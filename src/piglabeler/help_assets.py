from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from piglabeler.constants import CLASS_NAMES, HELP_EXAMPLES_PER_CLASS
from piglabeler.dataset import DatasetCatalog, PigTask
from piglabeler.rendering import _load_font


class HelpAssetBuilder:
    def __init__(self, state_dir: Path) -> None:
        self.cache_dir = state_dir / "render_cache" / "help"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.title_font = _load_font(26)
        self.body_font = _load_font(18)

    def build_examples(self, catalog: DatasetCatalog) -> Path:
        output_path = self.cache_dir / "class_examples.jpg"
        if output_path.exists():
            return output_path

        reference_tasks = self._select_reference_tasks(catalog)
        thumb_size = (220, 180)
        margin = 12
        header_height = 34
        columns = HELP_EXAMPLES_PER_CLASS
        rows = len(CLASS_NAMES)

        canvas = Image.new(
            "RGB",
            (
                margin + columns * (thumb_size[0] + margin),
                margin + rows * (header_height + thumb_size[1] + margin * 2),
            ),
            "#ffffff",
        )
        draw = ImageDraw.Draw(canvas)

        y = margin
        for class_id in sorted(CLASS_NAMES):
            tasks = reference_tasks.get(class_id, [])
            draw.rounded_rectangle(
                (0, y - 4, canvas.width, y + header_height),
                radius=0,
                fill="#f1f5f9",
            )
            draw.text(
                (margin, y + 5),
                f"{class_id} — {CLASS_NAMES[class_id]}",
                fill="#0f172a",
                font=self.title_font,
            )
            row_y = y + header_height + 4
            for column, task in enumerate(tasks):
                thumb = self._build_thumb(task, thumb_size)
                x = margin + column * (thumb_size[0] + margin)
                canvas.paste(thumb, (x, row_y))
            y += header_height + thumb_size[1] + margin * 2

        canvas.save(output_path, format="JPEG", quality=88)
        return output_path

    def _select_reference_tasks(self, catalog: DatasetCatalog) -> dict[int, list[PigTask]]:
        target_views = {
            task.camera_view
            for task in catalog.iter_annotatable_tasks()
            if task.camera_view is not None
        }
        candidates = sorted(
            catalog.labeled_reference_tasks(),
            key=lambda task: (
                0 if task.camera_view in target_views else 1,
                -task.bbox_area,
                task.image_id,
            ),
        )

        buckets: dict[int, list[PigTask]] = defaultdict(list)
        used_keys: set[tuple[int, str]] = set()
        for task in candidates:
            if task.class_id is None:
                continue
            if len(buckets[task.class_id]) >= HELP_EXAMPLES_PER_CLASS:
                continue
            view_key = (task.class_id, task.camera_view or "unknown")
            if view_key in used_keys and len(buckets[task.class_id]) < HELP_EXAMPLES_PER_CLASS - 1:
                continue
            buckets[task.class_id].append(task)
            used_keys.add(view_key)

        for class_id in CLASS_NAMES:
            buckets.setdefault(class_id, [])
        return dict(buckets)

    def _build_thumb(self, task: PigTask, size: tuple[int, int]) -> Image.Image:
        image = Image.open(task.image_path).convert("RGB")
        bx, by, bw, bh = task.bbox
        left = max(0, int(bx - bw * 0.15))
        top = max(0, int(by - bh * 0.15))
        right = min(image.width, int(bx + bw * 1.15))
        bottom = min(image.height, int(by + bh * 1.15))
        crop = image.crop((left, top, right, bottom))
        crop = ImageOps.contain(crop, (size[0], size[1] - 18))

        thumb = Image.new("RGB", size, "#ffffff")
        thumb.paste(crop, ((size[0] - crop.width) // 2, 0))
        draw = ImageDraw.Draw(thumb)
        draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=12, outline="#d8d8d8", width=2)
        draw.rectangle((0, size[1] - 18, size[0], size[1]), fill="#111827")
        label = f"{task.pen or '?'} {task.camera_view or '?'}"
        draw.text((6, size[1] - 16), label, fill="#ffffff", font=self.body_font)
        return thumb
