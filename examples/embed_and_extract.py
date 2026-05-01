from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from rise_machine_spirit.analysis import save_spectrogram_diff, segmental_snr_db, snr_db
from rise_machine_spirit.audio import read_lossless_audio
from rise_machine_spirit.stego import StegoParams, embed_png, extract_png


def pink_noise(seconds: float = 10.0, samplerate: int = 48_000, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(2026)
    samples = int(seconds * samplerate)
    freqs = np.fft.rfftfreq(samples, d=1.0 / samplerate)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    output = []
    for _channel in range(channels):
        spectrum = np.fft.rfft(rng.normal(size=samples)) * scale
        channel = np.fft.irfft(spectrum, n=samples)
        output.append(channel / np.max(np.abs(channel)) * 0.35)
    return np.stack(output, axis=1).astype(np.float64)


def make_payload(path: Path) -> None:
    image = Image.new("RGBA", (96, 96), (18, 18, 22, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 80, 80), outline=(245, 245, 245, 255), width=3)
    draw.line((20, 68, 42, 42, 56, 56, 76, 24), fill=(80, 220, 180, 255), width=4)
    image.save(path, "PNG", optimize=True)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    carrier = out_dir / "sample-carrier.wav"
    payload = out_dir / "sample-payload.png"
    stego = out_dir / "sample-stego.wav"
    recovered = out_dir / "sample-recovered.png"
    diff = out_dir / "sample-spectrogram-diff.png"

    sf.write(carrier, pink_noise(), 48_000, subtype="PCM_24")
    make_payload(payload)

    params = StegoParams(qim_floor=0.0012, iterations=3)
    embed_png(carrier, payload, stego, key="example-seed", params=params)
    extract_png(stego, recovered, key="example-seed", params=params)

    original, meta = read_lossless_audio(carrier)
    encoded, _ = read_lossless_audio(stego)
    save_spectrogram_diff(original, encoded, meta.samplerate, diff)
    print(f"SNR: {snr_db(original, encoded):.2f} dB")
    print(f"Segmental SNR: {segmental_snr_db(original, encoded):.2f} dB")
    print(f"Recovered matches payload: {recovered.read_bytes() == payload.read_bytes()}")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
