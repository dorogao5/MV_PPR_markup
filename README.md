# MV_PPR_markup

Telegram-бот для ручной разметки датасета Kaggle `Multi-view Pig Posture Recognition`.

Бот рассчитан на практический сценарий разметки:

- сам скачивает данные соревнования через `kaggle competitions download`;
- ищет все неразмеченные CSV-источники в `DATA_DIR`, а не только `test.csv`;
- показывает карточку задания: крупный crop цели и полный кадр с выделенным bbox;
- дополнительно присылает оригинальный кадр без bbox отдельной фотографией;
- умеет читать заранее подготовленную локально очередь с автопредсказанием и уверенностью;
- размечает через inline-кнопки с подтверждением `Confirm/Cancel`;
- сохраняет каждое действие сразу в файлы;
- считает статистику и умеет `undo`;
- генерирует help-референсы из уже размеченных train-сплитов;
- собирается и запускается в одном Docker-контейнере.

## Как бот понимает, что размечать

Бот просматривает все CSV в `DATA_DIR` и берет в работу те, где есть колонки:

- `row_id`
- `image_id`
- `width`
- `height`
- `bbox`

Если в источнике нет `class_id`, он считается неразмеченным и попадает в очередь на ручную разметку. Это позволяет автоматически подхватывать будущие файлы вроде `eval.csv`, `holdout.csv` и другие похожие сплиты.

В текущем архиве Kaggle основным неразмеченным источником является `test.csv`.

Если рядом есть готовый `prediction_queue.csv`, бот берет задачи в этом порядке. Это удобно для локальной приоритизации: сначала самые неуверенные автопредсказания, потом более уверенные.

## Что нужно перед запуском

1. Создать Telegram-бота через BotFather и получить токен.
2. Один раз открыть страницу соревнования в браузере и принять правила.
3. Получить Kaggle API credentials: `KAGGLE_USERNAME` и `KAGGLE_KEY`.
4. Скопировать `.env.example` в `.env` и заполнить значения.

Пример:

```bash
cp .env.example .env
```

## Запуск через Docker

Собрать и запустить контейнер:

```bash
docker compose up -d --build
```

После запуска открыть бота в Telegram и отправить `/start`.

## Проверка, что все работает

Проверить, что контейнер поднялся:

```bash
docker compose ps
```

Посмотреть логи:

```bash
docker compose logs -f
```

Если бот уже отвечает в Telegram, можно дополнительно проверить команды:

- `/start`
- `/next`
- `/help`
- `/stats`
- `/sources`
- `/undo`

## Остановка

Остановить контейнер:

```bash
docker compose stop
```

Полностью остановить и удалить контейнер:

```bash
docker compose down --remove-orphans
```

## Что бот записывает на диск

Все runtime-артефакты лежат в `STATE_DIR`:

- `annotations/annotation_events.jsonl` — append-only журнал действий;
- `annotations/current_annotations.csv` — актуальный snapshot ручных меток;
- `exports/<source>_manual_labels.csv` — исходный CSV, дополненный ручными метками;
- `exports/<source>_submission.csv` — Kaggle-ready файл `row_id,class_id`;
- `render_cache/` — кэш карточек заданий и help-изображений.

## Команды бота

- `/start` — приветствие и первая задача
- `/next` — показать следующую задачу
- `/help` — правила разметки и примеры классов
- `/stats` — прогресс по разметке
- `/sources` — найденные источники и их размеры
- `/undo` — откатить последнюю подтвержденную разметку

## Локальный запуск без Docker

Если данные уже лежат рядом с проектом:

```bash
uv sync
TELEGRAM_BOT_TOKEN=... DATA_DIR=. STATE_DIR=.state uv run pig-labeler-bot
```

## Локальная подготовка очереди из `probs/*.npy`

Сортировка по уверенности делается локально, а сервер потом только читает готовый CSV.

1. Установить `numpy` в локальную `.venv`, если его там еще нет:

```bash
uv pip install --python .venv/bin/python numpy
```

2. Сгенерировать очередь из `test.csv` и `probs/*.npy`:

```bash
PYTHONPATH=src .venv/bin/python -m piglabeler.prepare_queue \
  --csv test.csv \
  --output predictions/prediction_queue.csv \
  --strategy ensemble-mean
```

Что делает команда:

- берет `row_id` в порядке строк из `test.csv`;
- читает заранее сохраненные вероятности из `probs/`;
- строит автопредсказание;
- сортирует очередь от самой низкой уверенности к самой высокой;
- пишет готовый `predictions/prediction_queue.csv`, который бот потом использует без каких-либо вычислений на сервере.

Если хочешь строить очередь по одной модели, замени команду на `--strategy best-model --model swin_tta` или другую доступную модель.

При запуске бот автоматически подхватит:

- `DATA_DIR/prediction_queue.csv`
- `DATA_DIR/test_prediction_queue.csv`
- или `predictions/prediction_queue.csv` из репозитория

Если нужен конкретный путь, можно явно задать `PREDICTION_QUEUE_PATH`.

## Приоритетная очередь по disagreement

Если нужно размечать не просто low-confidence, а самые конфликтные строки, можно собрать полную очередь в штатном формате бота:

```bash
uv run python -m piglabeler.prepare_disagreement_queue \
  --csv server_dump/test.csv \
  --old-submission submissions/old/T1_smart_plus_new_all3_to34.csv \
  --old-submission submissions/old/T1_smart_plus_new_all3_no03_13_to4.csv \
  --output predictions/prediction_queue.csv
```

Что делает команда:

- считает базовое автопредсказание по среднему `effnet/maxvit/swin`;
- поднимает наверх строки, где старые сильные сабмиты не согласны между собой;
- затем поднимает случаи, где текущие модели не согласны друг с другом;
- отдельно усиливает пары классов `0/1` и `3/4`;
- записывает полноценный `prediction_queue.csv`, который бот читает напрямую по `priority_rank`.

Если `submissions/old/*.csv` уже лежат в репозитории, `--old-submission` можно не передавать: скрипт подхватит их автоматически.

## Локальная сборка финального submission

Когда ручная разметка уже выгружена с сервера, можно собрать итоговый сабмит локально:

```bash
PYTHONPATH=src .venv/bin/python -m piglabeler.build_submission \
  --csv test.csv \
  --probs-dir probs \
  --manual-labels current_annotations.csv \
  --output submissions/submission_blend_manual.csv
```

Что делает команда:

- усредняет вероятности из `probs/*.npy`;
- берет `argmax` как базовое автопредсказание;
- накладывает сверху ручные метки по `row_id`;
- пишет финальный Kaggle-файл `row_id,class_id`.

Если хочешь использовать уже готовый экспорт из бота, вместо `current_annotations.csv` можно передать `test_submission.csv` или `test_manual_labels.csv`.

Если нужен взвешенный бленд, добавь:

```bash
PYTHONPATH=src .venv/bin/python -m piglabeler.build_submission \
  --csv test.csv \
  --probs-dir probs \
  --strategy weighted-mean \
  --weight effnet_tta=2 \
  --weight maxvit_tta=1 \
  --weight swin_tta=1 \
  --manual-labels current_annotations.csv \
  --output submissions/submission_weighted_manual.csv
```

## Patch готового submission ручными метками

Если лучший baseline уже есть в виде готового `submission.csv`, можно просто наложить ручные метки без пересборки из `probs`:

```bash
PYTHONPATH=src .venv/bin/python -m piglabeler.patch_submission \
  --base-submission submissions/old/best_old.csv \
  --manual-labels server_dump/test_submission.csv \
  --output submissions/best_old_manual.csv
```

Команда:

- читает базовый `row_id,class_id`;
- заменяет совпавшие `row_id` на ручные метки;
- пишет новый patched submission.

## Замечания

- Для различения `Lateral_lying_left` и `Lateral_lying_right` бот показывает реальные train-примеры в `/help`.
- Контейнер работает через long polling и в `docker-compose.yml` уже настроен `restart: unless-stopped`.
- Если данные уже существуют в `DATA_DIR`, автоскачивание не перетирает их.
