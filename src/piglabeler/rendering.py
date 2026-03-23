from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from piglabeler.dataset import PigTask


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidate_paths:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


class TaskRenderer:
    def __init__(self, state_dir: Path) -> None:
        self.cache_dir = state_dir / "render_cache" / "tasks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.title_font = _load_font(30)
        self.body_font = _load_font(22)
        self.small_font = _load_font(18)

    def render_task(self, task: PigTask, siblings: list[PigTask]) -> Path:
        output_path = self.cache_dir / f"{task.row_id}.jpg"
        if output_path.exists():
            return output_path

        image = Image.open(task.image_path).convert("RGB")
        crop = self._make_crop(image, task)
        full = self._make_full_scene(image, task, siblings)

        canvas = Image.new("RGB", (1600, 980), "#f4f4f2")
        draw = ImageDraw.Draw(canvas)

        draw.text((60, 36), "Target crop", fill="#111111", font=self.title_font)
        draw.text((840, 36), "Full scene", fill="#111111", font=self.title_font)

        crop_panel = self._fit_panel(crop, (700, 700))
        full_panel = self._fit_panel(full, (700, 700))

        canvas.paste(crop_panel, (60, 110))
        canvas.paste(full_panel, (840, 110))

        footer_y = 845
        draw.rounded_rectangle((40, footer_y, 1560, 940), radius=18, fill="#ffffff")
        footer_lines = [
            f"source: {task.source_name}",
            f"row_id: {task.row_id}",
            f"image: {task.image_id}",
            f"bbox: [{task.bbox[0]:.1f}, {task.bbox[1]:.1f}, {task.bbox[2]:.1f}, {task.bbox[3]:.1f}]",
        ]
        for index, line in enumerate(footer_lines):
            draw.text(
                (70 + (index % 2) * 700, footer_y + 20 + (index // 2) * 34),
                line,
                fill="#222222",
                font=self.body_font,
            )

        canvas.save(output_path, format="JPEG", quality=90)
        return output_path

    def _make_crop(self, image: Image.Image, task: PigTask) -> Image.Image:
        bx, by, bw, bh = task.bbox
        pad_x = bw * 0.2
        pad_y = bh * 0.2
        left = max(0, int(math.floor(bx - pad_x)))
        top = max(0, int(math.floor(by - pad_y)))
        right = min(image.width, int(math.ceil(bx + bw + pad_x)))
        bottom = min(image.height, int(math.ceil(by + bh + pad_y)))

        crop = image.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        draw.rounded_rectangle(
            (
                bx - left,
                by - top,
                bx - left + bw,
                by - top + bh,
            ),
            radius=8,
            outline="#d7263d",
            width=6,
        )
        return crop

    def _make_full_scene(
        self,
        image: Image.Image,
        task: PigTask,
        siblings: list[PigTask],
    ) -> Image.Image:
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)

        for sibling in siblings:
            bx, by, bw, bh = sibling.bbox
            color = "#d7263d" if sibling.row_id == task.row_id else "#9ca3af"
            width = 7 if sibling.row_id == task.row_id else 3
            draw.rounded_rectangle(
                (bx, by, bx + bw, by + bh),
                radius=8,
                outline=color,
                width=width,
            )

        label = task.row_id.split("_")[-1]
        bx, by, _, _ = task.bbox
        text_box = (bx + 6, max(4, by - 34), bx + 130, max(34, by - 4))
        draw.rounded_rectangle(text_box, radius=8, fill="#d7263d")
        draw.text(
            (text_box[0] + 8, text_box[1] + 6),
            label,
            fill="#ffffff",
            font=self.small_font,
        )
        return overlay

    def _fit_panel(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        frame = Image.new("RGB", size, "#ffffff")
        fitted = ImageOps.contain(image, size)
        offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
        frame.paste(fitted, offset)
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=14, outline="#d8d8d8", width=2)
        return frame
