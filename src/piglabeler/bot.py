from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from piglabeler.constants import (
    CLASS_BUTTON_RU,
    CLASS_NAMES,
    CLASS_SHORT_RU,
    HELP_TEXT,
    MAX_RECENT_SKIPS,
    SESSION_TTL_MINUTES,
)
from piglabeler.dataset import DatasetCatalog, PigTask
from piglabeler.help_assets import HelpAssetBuilder
from piglabeler.rendering import TaskRenderer
from piglabeler.store import AnnotationStore

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionState:
    current_row_id: str | None = None
    pending_class_id: int | None = None
    current_message_ids: list[int] = field(default_factory=list)
    skipped_row_ids: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_SKIPS))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)


class PigLabelerBot:
    def __init__(
        self,
        *,
        token: str,
        catalog: DatasetCatalog,
        store: AnnotationStore,
        renderer: TaskRenderer,
        help_builder: HelpAssetBuilder,
    ) -> None:
        self.catalog = catalog
        self.store = store
        self.renderer = renderer
        self.help_builder = help_builder
        self.sessions: dict[int, SessionState] = {}

        self.application: Application = ApplicationBuilder().token(token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("next", self.next_task_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("sources", self.sources_command))
        self.application.add_handler(CommandHandler("undo", self.undo_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

    def run(self) -> None:
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        assert user is not None
        await update.effective_chat.send_message(
            "Бот готов к разметке.\n"
            "Команды: /next, /help, /stats, /sources, /undo\n"
            "Ниже отправляю первую задачу."
        )
        await self._send_task_for_user(
            chat_id=update.effective_chat.id,
            user_id=user.id,
            user_name=self._user_name(user),
            context=context,
        )

    async def next_task_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        assert user is not None
        await self._send_task_for_user(
            chat_id=update.effective_chat.id,
            user_id=user.id,
            user_name=self._user_name(user),
            context=context,
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._send_help(update.effective_chat.id, context)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_chat.send_message(self._build_stats_text())

    async def sources_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_chat.send_message(self._build_sources_text())

    async def undo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        assert user is not None
        result = await self.store.undo_last_for_user(
            annotator_id=user.id,
            annotator_name=self._user_name(user),
            catalog=self.catalog,
        )
        await update.effective_chat.send_message(result.message)
        if result.ok and result.row_id:
            session = self._get_session(user.id)
            session.current_row_id = result.row_id
            session.pending_class_id = None
            await self._send_task_for_user(
                chat_id=update.effective_chat.id,
                user_id=user.id,
                user_name=self._user_name(user),
                context=context,
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return

        user = query.from_user
        session = self._get_session(user.id)
        session.touch()
        data = query.data or ""

        if data.startswith("pick:"):
            class_id = int(data.split(":", 1)[1])
            session.pending_class_id = class_id
            task = self._current_task_for_session(session)
            if task is None:
                await query.answer("Текущая задача уже недоступна.", show_alert=True)
                return
            await query.edit_message_caption(
                caption=self._build_task_caption(task, pending_class_id=class_id),
                reply_markup=self._build_confirm_keyboard(class_id),
            )
            await query.answer("Выбор сохранён, осталось подтвердить.")
            return

        if data == "cancel":
            session.pending_class_id = None
            task = self._current_task_for_session(session)
            if task is None:
                await query.answer("Текущая задача уже недоступна.", show_alert=True)
                return
            await query.edit_message_caption(
                caption=self._build_task_caption(task),
                reply_markup=self._build_label_keyboard(),
            )
            await query.answer("Выбор отменён.")
            return

        if data.startswith("confirm:"):
            class_id = int(data.split(":", 1)[1])
            task = self._current_task_for_session(session)
            if task is None:
                await query.answer("Текущая задача уже недоступна.", show_alert=True)
                return
            await self.store.annotate(
                task,
                class_id,
                annotator_id=user.id,
                annotator_name=self._user_name(user),
                catalog=self.catalog,
            )
            session.current_row_id = None
            session.pending_class_id = None
            await query.answer("Разметка сохранена.")
            await self._delete_task_messages(
                chat_id=query.message.chat.id,
                session=session,
                context=context,
            )
            await self._send_task_for_user(
                chat_id=query.message.chat.id,
                user_id=user.id,
                user_name=self._user_name(user),
                context=context,
            )
            return

        if data == "skip":
            task = self._current_task_for_session(session)
            if task is not None:
                session.skipped_row_ids.append(task.row_id)
            session.current_row_id = None
            session.pending_class_id = None
            await query.answer("Задача пропущена.")
            await self._delete_task_messages(
                chat_id=query.message.chat.id,
                session=session,
                context=context,
            )
            await self._send_task_for_user(
                chat_id=query.message.chat.id,
                user_id=user.id,
                user_name=self._user_name(user),
                context=context,
            )
            return

        if data == "help":
            await query.answer()
            await self._send_help(query.message.chat.id, context)
            return

        if data == "stats":
            await query.answer()
            await context.bot.send_message(query.message.chat.id, self._build_stats_text())
            return

        if data == "undo":
            result = await self.store.undo_last_for_user(
                annotator_id=user.id,
                annotator_name=self._user_name(user),
                catalog=self.catalog,
            )
            await query.answer(result.message, show_alert=not result.ok)
            if result.ok and result.row_id:
                session.current_row_id = result.row_id
                session.pending_class_id = None
                await self._send_task_for_user(
                    chat_id=query.message.chat.id,
                    user_id=user.id,
                    user_name=self._user_name(user),
                    context=context,
                )
            return

    async def _send_task_for_user(
        self,
        *,
        chat_id: int,
        user_id: int,
        user_name: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        session = self._get_session(user_id)
        await self._delete_task_messages(chat_id=chat_id, session=session, context=context)
        task = self._pick_next_task(session, user_id)
        if task is None:
            await context.bot.send_message(
                chat_id,
                "Неразмеченных задач больше нет. Экспорт уже лежит в STATE_DIR/exports.",
            )
            return

        siblings = self.catalog.get_image_tasks(task.source_name, task.image_id)
        image_path = self.renderer.render_task(task, siblings)
        with task.image_path.open("rb") as original_handle:
            original_message = await context.bot.send_photo(
                chat_id=chat_id,
                photo=original_handle,
                caption=self._build_original_caption(task),
            )
        with image_path.open("rb") as handle:
            task_message = await context.bot.send_photo(
                chat_id=chat_id,
                photo=handle,
                caption=self._build_task_caption(task, pending_class_id=session.pending_class_id),
                reply_markup=(
                    self._build_confirm_keyboard(session.pending_class_id)
                    if session.pending_class_id is not None
                    else self._build_label_keyboard()
                ),
            )
        session.current_message_ids = [original_message.message_id, task_message.message_id]
        session.touch()

    async def _send_help(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_image = self.help_builder.build_examples(self.catalog)
        with help_image.open("rb") as handle:
            await context.bot.send_photo(chat_id=chat_id, photo=handle, caption=HELP_TEXT)

    def _get_session(self, user_id: int) -> SessionState:
        self._prune_sessions()
        session = self.sessions.get(user_id)
        if session is None:
            session = SessionState()
            self.sessions[user_id] = session
        session.touch()
        return session

    def _prune_sessions(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(minutes=SESSION_TTL_MINUTES)
        stale_user_ids = [
            user_id
            for user_id, session in self.sessions.items()
            if session.updated_at < cutoff
        ]
        for user_id in stale_user_ids:
            self.sessions.pop(user_id, None)

    def _claimed_row_ids(self, *, exclude_user_id: int | None = None) -> set[str]:
        claimed = set()
        for user_id, session in self.sessions.items():
            if exclude_user_id is not None and user_id == exclude_user_id:
                continue
            if session.current_row_id:
                claimed.add(session.current_row_id)
        return claimed

    def _pick_next_task(self, session: SessionState, user_id: int) -> PigTask | None:
        existing = self._current_task_for_session(session)
        if existing is not None:
            return existing

        claimed = self._claimed_row_ids(exclude_user_id=user_id)
        skipped = set(session.skipped_row_ids)

        for task in self.catalog.iter_annotatable_tasks():
            if self.store.is_annotated(task.row_id):
                continue
            if task.row_id in claimed:
                continue
            if task.row_id in skipped:
                continue
            session.current_row_id = task.row_id
            session.pending_class_id = None
            session.touch()
            return task

        if skipped:
            session.skipped_row_ids.clear()
            return self._pick_next_task(session, user_id)

        return None

    def _current_task_for_session(self, session: SessionState) -> PigTask | None:
        if not session.current_row_id:
            return None
        if self.store.is_annotated(session.current_row_id):
            session.current_row_id = None
            session.pending_class_id = None
            return None
        return self.catalog.get_task(session.current_row_id)

    def _build_task_caption(
        self,
        task: PigTask,
        *,
        pending_class_id: int | None = None,
    ) -> str:
        source_totals = self.catalog.source_totals()
        done_by_source = self.store.count_by_source()
        total_done = len(self.store.current_annotations)
        total_all = len(self.catalog.annotatable_row_ids)
        source_done = done_by_source.get(task.source_name, 0)
        source_total = source_totals.get(task.source_name, 0)

        lines = [
            "Выбери позу для свиньи в красном bbox.",
            f"Источник: {task.source_name}",
            f"row_id: {task.row_id}",
            f"Кадр: {task.image_id}",
            f"Прогресс источника: {source_done}/{source_total}",
            f"Общий прогресс: {total_done}/{total_all}",
        ]
        if task.camera_view:
            lines.append(f"Камера: {task.pen} / {task.camera_view}")
        prediction = self.catalog.prediction_for(task.row_id)
        if prediction is not None:
            lines.append(
                "Автопредсказание: "
                f"{prediction.predicted_class_id} — "
                f"{CLASS_SHORT_RU[prediction.predicted_class_id]} "
                f"({CLASS_NAMES[prediction.predicted_class_id]})"
            )
            lines.append(
                f"Источник подсказки: {prediction.model_name}, "
                f"уверенность {prediction.confidence * 100:.1f}%"
            )
        if pending_class_id is not None:
            lines.append(
                f"Выбран класс: {pending_class_id} — {CLASS_SHORT_RU[pending_class_id]} "
                f"({CLASS_NAMES[pending_class_id]})"
            )
            lines.append("Подтверди выбор или нажми «Отмена».")
        else:
            lines.append("Сначала нажми класс, потом «Подтвердить».")
        return "\n".join(lines)

    def _build_original_caption(self, task: PigTask) -> str:
        return (
            "Оригинальный кадр без bbox.\n"
            f"Кадр: {task.image_id}\n"
            "Открой фото отдельно, если нужно приблизить детали."
        )

    def _build_label_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(CLASS_BUTTON_RU[0], callback_data="pick:0"),
                    InlineKeyboardButton(CLASS_BUTTON_RU[1], callback_data="pick:1"),
                ],
                [
                    InlineKeyboardButton(CLASS_BUTTON_RU[2], callback_data="pick:2"),
                    InlineKeyboardButton(CLASS_BUTTON_RU[3], callback_data="pick:3"),
                ],
                [InlineKeyboardButton(CLASS_BUTTON_RU[4], callback_data="pick:4")],
                [
                    InlineKeyboardButton("Пропустить", callback_data="skip"),
                    InlineKeyboardButton("Помощь", callback_data="help"),
                    InlineKeyboardButton("Статистика", callback_data="stats"),
                ],
                [InlineKeyboardButton("Отменить последнюю", callback_data="undo")],
            ]
        )

    def _build_confirm_keyboard(self, class_id: int) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"Подтвердить: {class_id} — {CLASS_SHORT_RU[class_id]}",
                        callback_data=f"confirm:{class_id}",
                    )
                ],
                [InlineKeyboardButton("Отмена", callback_data="cancel")],
                [
                    InlineKeyboardButton("Помощь", callback_data="help"),
                    InlineKeyboardButton("Статистика", callback_data="stats"),
                ],
            ]
        )

    def _build_stats_text(self) -> str:
        total_done = len(self.store.current_annotations)
        total_all = len(self.catalog.annotatable_row_ids)
        class_counts = self.store.count_by_class()
        source_totals = self.catalog.source_totals()
        source_done = self.store.count_by_source()

        lines = [
            "Статистика разметки:",
            f"Всего подтверждено: {total_done}/{total_all}",
            "",
            "По источникам:",
        ]
        for source_name, total in sorted(source_totals.items()):
            lines.append(f"- {source_name}: {source_done.get(source_name, 0)}/{total}")

        lines.append("")
        lines.append("По классам:")
        for class_id, class_name in CLASS_NAMES.items():
            lines.append(f"- {class_id} {class_name}: {class_counts.get(class_id, 0)}")
        return "\n".join(lines)

    def _build_sources_text(self) -> str:
        annotatable = set(self.catalog.source_totals())
        lines = ["Найденные источники:"]
        for source_name, summary in sorted(self.catalog.sources.items()):
            mode = "annotate" if source_name in annotatable else "reference"
            lines.append(
                f"- {source_name}: rows={summary.total_rows}, images={summary.total_images}, "
                f"has_labels={summary.has_labels}, mode={mode}"
            )
        return "\n".join(lines)

    async def _delete_task_messages(
        self,
        *,
        chat_id: int,
        session: SessionState,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not session.current_message_ids:
            return

        for message_id in session.current_message_ids:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception:  # pragma: no cover - Telegram errors are non-fatal here
                LOGGER.debug("Failed to delete task message %s", message_id, exc_info=True)
        session.current_message_ids.clear()

    def _user_name(self, user) -> str:
        if user.username:
            return f"@{user.username}"
        full_name = " ".join(part for part in [user.first_name, user.last_name] if part).strip()
        return full_name or str(user.id)
