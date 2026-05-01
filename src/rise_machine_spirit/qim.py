from __future__ import annotations

import numpy as np


def encode_qim(value: float, bit: int, delta: float) -> float:
    if delta <= 0:
        raise ValueError("QIM delta must be positive.")
    center = int(np.floor(value / delta))
    candidates = []
    for n in range(center - 3, center + 4):
        if (n & 1) == int(bit):
            candidates.append(n * delta)
    return min(candidates, key=lambda candidate: abs(candidate - value))


def decode_qim(value: float, delta: float) -> tuple[int, float]:
    if delta <= 0:
        raise ValueError("QIM delta must be positive.")
    quant_index = int(np.rint(value / delta))
    bit = quant_index & 1
    residual = abs((value / delta) - quant_index)
    confidence = max(0.0, 1.0 - min(1.0, residual * 2.0))
    return bit, confidence
