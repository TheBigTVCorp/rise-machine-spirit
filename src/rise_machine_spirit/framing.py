from __future__ import annotations

import numpy as np


def overlap_window(frame_size: int) -> np.ndarray:
    return np.ones(frame_size, dtype=np.float64)


def frame_count(sample_count: int, frame_size: int, hop_size: int) -> int:
    if sample_count <= frame_size:
        return 1
    return int(np.ceil((sample_count - frame_size) / hop_size)) + 1


def padded_length(sample_count: int, frame_size: int, hop_size: int) -> int:
    return (frame_count(sample_count, frame_size, hop_size) - 1) * hop_size + frame_size


def padded_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    target = padded_length(audio.shape[0], frame_size, hop_size)
    if target == audio.shape[0]:
        return audio.copy()
    pad = np.zeros((target - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.vstack([audio, pad])
