from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    noise = reference - candidate
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def segmental_snr_db(reference: np.ndarray, candidate: np.ndarray, frame_size: int = 4096) -> float:
    values = []
    for start in range(0, reference.shape[0], frame_size):
        ref = reference[start : start + frame_size]
        cand = candidate[start : start + frame_size]
        if ref.size == 0:
            continue
        values.append(snr_db(ref, cand))
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("inf")


def save_spectrogram_diff(
    reference: np.ndarray,
    candidate: np.ndarray,
    samplerate: int,
    output: str | Path,
) -> None:
    mono_ref = np.mean(reference, axis=1) if reference.ndim == 2 else reference
    mono_cand = np.mean(candidate, axis=1) if candidate.ndim == 2 else candidate
    freqs, times, spec_ref = signal.spectrogram(mono_ref, samplerate, nperseg=2048, noverlap=1024)
    _freqs, _times, spec_cand = signal.spectrogram(
        mono_cand, samplerate, nperseg=2048, noverlap=1024
    )
    diff = 10.0 * np.log10(np.maximum(spec_cand, 1e-18)) - 10.0 * np.log10(
        np.maximum(spec_ref, 1e-18)
    )

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    mesh = ax.pcolormesh(times, freqs, diff, shading="auto", cmap="coolwarm", vmin=-24, vmax=24)
    ax.set_title("Stego minus carrier spectrogram difference")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.savefig(output, dpi=150)
    plt.close(fig)
