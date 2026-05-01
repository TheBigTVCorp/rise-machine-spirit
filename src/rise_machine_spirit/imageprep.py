from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    size_bytes: int
    quality: int
    budget_bytes: int
    dimensions: tuple[int, int]


def _save_jpeg(path: Path, image: Image.Image, quality: int) -> None:
    try:
        image.save(path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=1)
    except OSError:
        try:
            image.save(
                path, "JPEG", quality=quality, optimize=True, progressive=False, subsampling=1
            )
        except OSError:
            image.save(
                path, "JPEG", quality=quality, optimize=False, progressive=False, subsampling=1
            )


def prepare_jpeg_to_budget(
    input_path: str | Path,
    output_path: str | Path,
    budget_bytes: int,
    *,
    quality_min: int = 30,
    quality_max: int = 92,
) -> PreparedImage:
    if budget_bytes <= 0:
        raise ValueError("Image byte budget must be positive.")
    if quality_min < 1 or quality_max > 95 or quality_min > quality_max:
        raise ValueError("JPEG quality range must be within 1..95 and min <= max.")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path).convert("RGB")

    working = image
    for _scale_attempt in range(12):
        for quality in range(quality_max, quality_min - 1, -1):
            _save_jpeg(output_path, working, quality)
            size = output_path.stat().st_size
            if size <= budget_bytes:
                return PreparedImage(output_path, size, quality, budget_bytes, working.size)
        next_size = (max(64, int(working.width * 0.85)), max(64, int(working.height * 0.85)))
        if next_size == working.size:
            break
        working = image.resize(next_size, Image.Resampling.LANCZOS)

    _save_jpeg(output_path, working, quality_min)
    size = output_path.stat().st_size
    raise ValueError(
        f"Could not fit image under {budget_bytes} bytes; q{quality_min} at {working.size} produced {size} bytes."
    )
