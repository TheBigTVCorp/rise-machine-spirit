from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt

from .audio import AudioMeta, read_lossless_audio, write_lossless_audio
from .exceptions import CapacityError, PayloadFormatError
from .framing import frame_count, overlap_window, padded_audio
from .payload import (
    HEADER_SIZE,
    bits_to_bytes,
    bytes_to_bits,
    parse_header,
    parse_packet,
    serialize_image,
)
from .psycho import masking_deltas
from .qim import decode_qim, encode_qim

DEFAULT_FRAME_SIZE = 4096
DEFAULT_OVERLAP = 0.5
DEFAULT_WAVELET = "bior4.4"
DEFAULT_LEVEL = 5
DEFAULT_QIM_FLOOR = 7.5e-4
EMBED_LEVELS = (5, 4, 3)


@dataclass(frozen=True)
class StegoParams:
    frame_size: int = DEFAULT_FRAME_SIZE
    overlap: float = DEFAULT_OVERLAP
    wavelet: str = DEFAULT_WAVELET
    level: int = DEFAULT_LEVEL
    qim_floor: float = DEFAULT_QIM_FLOOR
    iterations: int = 2
    redundancy: int = 5

    @property
    def hop_size(self) -> int:
        return int(round(self.frame_size * (1.0 - self.overlap)))


@dataclass(frozen=True)
class EmbedResult:
    capacity_bytes: int
    payload_bytes: int
    sample_rate: int
    channels: int
    clipped_samples: int = 0
    peak_before_clip: float = 0.0


@dataclass(frozen=True)
class Position:
    frame: int
    channel: int
    level: int
    coeff_index: int

    def label(self) -> str:
        return f"frame={self.frame}/ch={self.channel}/cD{self.level}"


def _seed_from_key(key: str) -> int:
    import hashlib

    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"rms-qim").digest()
    return int.from_bytes(digest, "big", signed=False)


def _coeff_template(frame_size: int, wavelet: str, level: int) -> dict[int, int]:
    coeffs = pywt.wavedec(
        np.zeros(frame_size, dtype=np.float64), wavelet, mode="periodization", level=level
    )
    return {5: len(coeffs[1]), 4: len(coeffs[2]), 3: len(coeffs[3])}


def _detail_array(coeffs: list[np.ndarray], detail_level: int) -> np.ndarray:
    index_by_level = {5: 1, 4: 2, 3: 3}
    return coeffs[index_by_level[detail_level]]


def _candidate_count(samples: int, channels: int, params: StegoParams) -> int:
    sizes = _coeff_template(params.frame_size, params.wavelet, params.level)
    return (
        frame_count(samples, params.frame_size, params.hop_size)
        * channels
        * sum(sizes[level] for level in EMBED_LEVELS)
    )


def _effective_capacity_bytes(samples: int, channels: int, params: StegoParams) -> int:
    return (_candidate_count(samples, channels, params) // max(1, params.redundancy)) // 8


def estimate_capacity(
    carrier: str | Path | np.ndarray,
    *,
    channels: int | None = None,
    params: StegoParams | None = None,
) -> int:
    params = params or StegoParams()
    if isinstance(carrier, np.ndarray):
        if carrier.ndim == 1:
            samples = carrier.shape[0]
            channel_count = channels or 1
        else:
            samples = carrier.shape[0]
            channel_count = channels or carrier.shape[1]
    else:
        audio, _meta = read_lossless_audio(carrier)
        samples = audio.shape[0]
        channel_count = audio.shape[1]
    return _effective_capacity_bytes(samples, channel_count, params)


def _ordinal_to_position(
    ordinal: int, channels: int, sizes: dict[int, int], params: StegoParams
) -> Position:
    per_channel = sum(sizes[level] for level in EMBED_LEVELS)
    per_frame = channels * per_channel
    frame = ordinal // per_frame
    rem = ordinal % per_frame
    channel = rem // per_channel
    coeff_rem = rem % per_channel
    for level in EMBED_LEVELS:
        count = sizes[level]
        if coeff_rem < count:
            return Position(
                frame=int(frame), channel=int(channel), level=level, coeff_index=int(coeff_rem)
            )
        coeff_rem -= count
    raise AssertionError("candidate ordinal did not map to a position")


def _positions(
    total: int, needed_bits: int, channels: int, key: str, params: StegoParams
) -> list[Position]:
    if needed_bits > total:
        raise CapacityError(total // 8, (needed_bits + 7) // 8)
    rng = np.random.default_rng(_seed_from_key(key))
    order = rng.permutation(total)[:needed_bits]
    sizes = _coeff_template(params.frame_size, params.wavelet, params.level)
    return [_ordinal_to_position(int(item), channels, sizes, params) for item in order]


def _decompose_frame(frame: np.ndarray, params: StegoParams) -> list[np.ndarray]:
    return pywt.wavedec(frame, params.wavelet, mode="periodization", level=params.level)


def _reconstruct_frame(coeffs: list[np.ndarray], params: StegoParams) -> np.ndarray:
    return pywt.waverec(coeffs, params.wavelet, mode="periodization")[: params.frame_size]


def _collect_position_map(
    positions: Iterable[Position], bits: np.ndarray
) -> dict[tuple[int, int], list[tuple[Position, int]]]:
    grouped: dict[tuple[int, int], list[tuple[Position, int]]] = {}
    for position, bit in zip(positions, bits, strict=True):
        grouped.setdefault((position.frame, position.channel), []).append((position, int(bit)))
    return grouped


def _apply_bits(
    audio: np.ndarray,
    bits: np.ndarray,
    positions: list[Position],
    meta: AudioMeta,
    params: StegoParams,
) -> np.ndarray:
    working = padded_audio(audio, params.frame_size, params.hop_size)
    window = overlap_window(params.frame_size)
    grouped = _collect_position_map(positions, bits)

    for _iteration in range(max(1, params.iterations)):
        out = np.zeros_like(working)
        norm = np.zeros((working.shape[0], 1), dtype=np.float64)
        for frame_index in range(frame_count(audio.shape[0], params.frame_size, params.hop_size)):
            start = frame_index * params.hop_size
            stop = start + params.frame_size
            norm[start:stop, 0] += window**2
            for channel in range(working.shape[1]):
                frame = working[start:stop, channel] * window
                coeffs = _decompose_frame(frame, params)
                entries = grouped.get((frame_index, channel), [])
                if entries:
                    deltas = masking_deltas(frame, meta.samplerate, qim_floor=params.qim_floor)
                    for position, bit in entries:
                        detail = _detail_array(coeffs, position.level)
                        detail[position.coeff_index] = encode_qim(
                            float(detail[position.coeff_index]), bit, deltas[position.level]
                        )
                reconstructed = _reconstruct_frame(coeffs, params)
                out[start:stop, channel] += reconstructed * window
        working = out / np.maximum(norm, 1e-12)
    return working[: audio.shape[0]]


def _read_bits(
    audio: np.ndarray,
    bit_count: int,
    positions: list[Position],
    meta: AudioMeta,
    params: StegoParams,
) -> tuple[np.ndarray, list[tuple[Position, float]]]:
    padded = padded_audio(audio, params.frame_size, params.hop_size)
    window = overlap_window(params.frame_size)
    grouped: dict[tuple[int, int], list[tuple[int, Position]]] = {}
    for bit_index, position in enumerate(positions):
        grouped.setdefault((position.frame, position.channel), []).append((bit_index, position))

    bits = np.zeros(bit_count, dtype=np.uint8)
    confidence: list[tuple[Position, float]] = []
    for (frame_index, channel), entries in grouped.items():
        start = frame_index * params.hop_size
        stop = start + params.frame_size
        frame = padded[start:stop, channel] * window
        coeffs = _decompose_frame(frame, params)
        deltas = masking_deltas(frame, meta.samplerate, qim_floor=params.qim_floor)
        for bit_index, position in entries:
            detail = _detail_array(coeffs, position.level)
            bit, conf = decode_qim(float(detail[position.coeff_index]), deltas[position.level])
            bits[bit_index] = bit
            confidence.append((position, conf))
    return bits, confidence


def _suspect_frames(confidence: list[tuple[Position, float]], limit: int = 12) -> list[str]:
    worst = sorted(confidence, key=lambda item: item[1])[:limit]
    labels: list[str] = []
    for position, _conf in worst:
        label = position.label()
        if label not in labels:
            labels.append(label)
    return labels


def _repeat_bits(bits: np.ndarray, redundancy: int) -> np.ndarray:
    redundancy = max(1, int(redundancy))
    if redundancy == 1:
        return bits
    return np.repeat(bits, redundancy).astype(np.uint8)


def _collapse_bits(
    bits: np.ndarray, confidence: list[tuple[Position, float]], redundancy: int
) -> tuple[np.ndarray, list[tuple[Position, float]]]:
    redundancy = max(1, int(redundancy))
    if redundancy == 1:
        return bits, confidence
    if bits.size % redundancy:
        raise PayloadFormatError("Recovered repeated bit stream has an invalid length.")

    collapsed = np.zeros(bits.size // redundancy, dtype=np.uint8)
    collapsed_confidence: list[tuple[Position, float]] = []
    for start in range(0, bits.size, redundancy):
        group = bits[start : start + redundancy]
        conf_group = confidence[start : start + redundancy]
        one_weight = sum(
            conf for bit, (_position, conf) in zip(group, conf_group, strict=True) if bit
        )
        zero_weight = sum(
            conf for bit, (_position, conf) in zip(group, conf_group, strict=True) if not bit
        )
        collapsed[start // redundancy] = 1 if one_weight >= zero_weight else 0
        collapsed_confidence.append(min(conf_group, key=lambda item: item[1]))
    return collapsed, collapsed_confidence


def embed_png(
    carrier: str | Path,
    payload: str | Path,
    output: str | Path,
    *,
    key: str,
    params: StegoParams | None = None,
) -> EmbedResult:
    params = params or StegoParams()
    audio, meta = read_lossless_audio(carrier)
    packet = serialize_image(payload, key=key)
    bits = bytes_to_bits(packet)
    total_candidates = _candidate_count(audio.shape[0], audio.shape[1], params)
    encoded_bits = _repeat_bits(bits, params.redundancy)
    if encoded_bits.size > total_candidates:
        raise CapacityError(
            _effective_capacity_bytes(audio.shape[0], audio.shape[1], params), len(packet)
        )
    positions = _positions(total_candidates, encoded_bits.size, audio.shape[1], key, params)
    stego = _apply_bits(audio, encoded_bits, positions, meta, params)
    clip = write_lossless_audio(output, stego, meta)
    return EmbedResult(
        capacity_bytes=_effective_capacity_bytes(audio.shape[0], audio.shape[1], params),
        payload_bytes=len(packet),
        sample_rate=meta.samplerate,
        channels=meta.channels,
        clipped_samples=clip.clipped_samples,
        peak_before_clip=clip.peak_before_clip,
    )


def extract_png(
    stego: str | Path,
    output: str | Path,
    *,
    key: str,
    params: StegoParams | None = None,
) -> None:
    packet = extract_image(stego, key=key, params=params)
    Path(output).write_bytes(packet.image_bytes)


def extract_image(
    stego: str | Path,
    *,
    key: str,
    params: StegoParams | None = None,
):
    params = params or StegoParams()
    audio, meta = read_lossless_audio(stego)
    total_candidates = _candidate_count(audio.shape[0], audio.shape[1], params)
    header_bits = HEADER_SIZE * 8
    encoded_header_bits = header_bits * max(1, params.redundancy)
    if total_candidates < encoded_header_bits:
        raise PayloadFormatError("Carrier is too short to contain a payload header.")

    header_positions = _positions(
        total_candidates, encoded_header_bits, audio.shape[1], key, params
    )
    header_repeated, header_repeated_confidence = _read_bits(
        audio, encoded_header_bits, header_positions, meta, params
    )
    header_array, header_confidence = _collapse_bits(
        header_repeated, header_repeated_confidence, params.redundancy
    )
    header_bytes = bits_to_bytes(header_array)
    header = parse_header(header_bytes)
    total_bytes = HEADER_SIZE + header.payload_length
    total_bits = total_bytes * 8
    encoded_total_bits = total_bits * max(1, params.redundancy)
    positions = _positions(total_candidates, encoded_total_bits, audio.shape[1], key, params)
    repeated_bits, repeated_confidence = _read_bits(
        audio, encoded_total_bits, positions, meta, params
    )
    bits, confidence = _collapse_bits(repeated_bits, repeated_confidence, params.redundancy)
    packet_bytes = bits_to_bytes(bits)
    suspect = _suspect_frames(header_confidence + confidence)
    packet = parse_packet(packet_bytes, key=key, suspect_frames=suspect)
    return packet
