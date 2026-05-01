from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageFilter, ImageOps


def image_to_spectrogram_wave(
    image_bytes: bytes,
    output: str | Path,
    *,
    samplerate: int = 48_000,
    duration: float = 12.0,
    min_freq: float = 1_000.0,
    max_freq: float = 20_000.0,
    fft_size: int = 2048,
    hop_size: int = 512,
    style: str = "image",
) -> Path:
    """Render image bytes into a WAV whose spectrogram displays the image."""

    output = Path(output)
    frame_count = max(8, int(duration * samplerate / hop_size))
    usable_bins = np.fft.rfftfreq(fft_size, 1.0 / samplerate)
    band = np.where((usable_bins >= min_freq) & (usable_bins <= max_freq))[0]
    if band.size < 16:
        raise ValueError("Frequency range is too small for a reveal spectrogram.")

    with Image.open(BytesIO(image_bytes)) as image:
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        if style == "edges":
            edges = gray.filter(ImageFilter.FIND_EDGES)
            gray = ImageOps.autocontrast(edges)
        elif style == "hybrid":
            edges = ImageOps.autocontrast(gray.filter(ImageFilter.FIND_EDGES))
            gray = Image.blend(gray, edges, 0.65)
            gray = ImageOps.autocontrast(gray)
        elif style != "image":
            raise ValueError("Reveal style must be image, hybrid, or edges.")
        gray.thumbnail((frame_count, band.size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (frame_count, band.size), 0)
        x = (frame_count - gray.width) // 2
        y = (band.size - gray.height) // 2
        canvas.paste(gray, (x, y))
        # Spectrograms put low frequencies at the bottom; image rows start at top.
        pixels = np.asarray(canvas, dtype=np.float64)[::-1, :] / 255.0

    spectra = np.zeros((frame_count, fft_size // 2 + 1), dtype=np.complex128)
    # Gamma lift makes midtones visible without clipping the loudest parts.
    magnitude = pixels**0.85 if style in {"edges", "hybrid"} else pixels**1.4
    phases = np.exp(1j * np.linspace(0.0, np.pi, band.size, endpoint=False))
    spectra[:, band] = (magnitude.T * phases[None, :]) * 0.08

    window = np.hanning(fft_size)
    samples = np.zeros((frame_count - 1) * hop_size + fft_size, dtype=np.float64)
    norm = np.zeros_like(samples)
    for index in range(frame_count):
        frame = np.fft.irfft(spectra[index], n=fft_size)
        start = index * hop_size
        stop = start + fft_size
        samples[start:stop] += frame * window
        norm[start:stop] += window**2
    samples = samples / np.maximum(norm, 1e-8)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.85
    stereo = np.column_stack([samples, samples])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, stereo, samplerate, subtype="PCM_24")
    return output
