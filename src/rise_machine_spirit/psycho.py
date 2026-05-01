from __future__ import annotations

import numpy as np

DETAIL_BANDS = {
    3: (1 / 16, 1 / 8),
    4: (1 / 32, 1 / 16),
    5: (1 / 64, 1 / 32),
}


def hz_to_bark(freq_hz: np.ndarray) -> np.ndarray:
    return 13.0 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500.0) ** 2)


def spectral_flatness(power: np.ndarray) -> float:
    positive = np.maximum(power, 1e-18)
    return float(np.exp(np.mean(np.log(positive))) / np.mean(positive))


def masking_deltas(
    frame: np.ndarray,
    samplerate: int,
    *,
    qim_floor: float,
    qim_ceiling: float = 0.01,
) -> dict[int, float]:
    """Return one QIM step for each DWT detail level.

    This is intentionally a compact psychoacoustic model: energy is grouped
    into Bark bands, adjusted by spectral flatness, and mapped onto the DWT
    detail bands used for embedding.
    """

    if frame.size == 0:
        return {level: qim_floor for level in DETAIL_BANDS}

    window = np.hanning(frame.size)
    spectrum = np.fft.rfft(frame * window)
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / samplerate)
    power = (np.abs(spectrum) / max(1, frame.size)) ** 2
    flatness = spectral_flatness(power)
    barks = hz_to_bark(freqs)
    deltas: dict[int, float] = {}

    for level, (lo_frac, hi_frac) in DETAIL_BANDS.items():
        lo_hz = samplerate * lo_frac
        hi_hz = samplerate * hi_frac
        band = (freqs >= lo_hz) & (freqs < hi_hz)
        if not np.any(band):
            deltas[level] = qim_floor
            continue

        band_power = power[band]
        band_bark = barks[band]
        bark_bins = np.floor(band_bark).astype(int)
        thresholds = []
        for bark in np.unique(bark_bins):
            values = band_power[bark_bins == bark]
            if values.size:
                thresholds.append(np.sqrt(np.percentile(values, 75)))
        threshold = float(np.mean(thresholds)) if thresholds else 0.0
        tonal_discount = 0.45 + 0.55 * min(1.0, flatness)
        delta = 0.6 * threshold * tonal_discount
        deltas[level] = float(np.clip(delta, qim_floor, qim_ceiling))

    return deltas
