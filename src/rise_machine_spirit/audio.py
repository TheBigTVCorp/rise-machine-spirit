from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .exceptions import UnsupportedCarrierError

LOSSY_EXTENSIONS = {".mp3", ".aac", ".m4a", ".opus", ".ogg", ".wma"}
LOSSLESS_EXTENSIONS = {".wav", ".flac"}


@dataclass(frozen=True)
class AudioMeta:
    samplerate: int
    channels: int
    format: str
    subtype: str


@dataclass(frozen=True)
class ClipReport:
    clipped_samples: int
    peak_before_clip: float

    @property
    def clipped(self) -> bool:
        return self.clipped_samples > 0


def assert_lossless_path(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix in LOSSY_EXTENSIONS:
        raise UnsupportedCarrierError(
            f"{path} is a lossy carrier. MP3/AAC/Opus-style codecs alter samples "
            "and will destroy a DWT/QIM payload; use WAV or FLAC."
        )
    if suffix not in LOSSLESS_EXTENSIONS:
        raise UnsupportedCarrierError(f"{path} is unsupported; only WAV and FLAC are accepted.")


def read_lossless_audio(path: str | Path) -> tuple[np.ndarray, AudioMeta]:
    assert_lossless_path(path)
    with sf.SoundFile(path) as sound:
        if sound.format not in {"WAV", "WAVEX", "FLAC"}:
            raise UnsupportedCarrierError(
                f"{path} is {sound.format}, not WAV or FLAC. Lossy carriers are refused."
            )
        audio = sound.read(dtype="float64", always_2d=True)
        meta = AudioMeta(
            samplerate=sound.samplerate,
            channels=sound.channels,
            format=sound.format,
            subtype=sound.subtype,
        )
    return audio, meta


def clip_report(audio: np.ndarray) -> ClipReport:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped_samples = int(np.count_nonzero((audio < -1.0) | (audio > 1.0)))
    return ClipReport(clipped_samples=clipped_samples, peak_before_clip=peak)


def write_lossless_audio(path: str | Path, audio: np.ndarray, meta: AudioMeta) -> ClipReport:
    assert_lossless_path(path)
    output_format = "FLAC" if Path(path).suffix.lower() == ".flac" else "WAV"
    subtype = meta.subtype
    if output_format == "FLAC" and subtype.startswith("PCM_U"):
        subtype = "PCM_16"
    report = clip_report(audio)
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(path, clipped, meta.samplerate, format=output_format, subtype=subtype)
    return report
