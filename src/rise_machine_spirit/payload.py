from __future__ import annotations

import hashlib
import secrets
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from PIL import Image

from .exceptions import AuthenticationError, PayloadFormatError

MAGIC = b"RMSIMG2\0"
VERSION = 2
MODE_BYTES = 16
SALT_BYTES = 16
NONCE_BYTES = 12
TAG_SIZE = 16
KDF_ITERATIONS = 600_000
HEADER_STRUCT = struct.Struct(f">8sBII B {MODE_BYTES}s Q {SALT_BYTES}s {NONCE_BYTES}s")
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True)
class PayloadHeader:
    width: int
    height: int
    mode: str
    payload_length: int
    salt: bytes
    nonce: bytes


@dataclass(frozen=True)
class PayloadPacket:
    header: PayloadHeader
    image_bytes: bytes
    auth_tag: bytes

    @property
    def png_bytes(self) -> bytes:
        return self.image_bytes


def _derive_key(key: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        key.encode("utf-8"),
        salt,
        KDF_ITERATIONS,
        dklen=32,
    )


def serialize_image(path: str | Path, *, key: str) -> bytes:
    path = Path(path)
    image_bytes = path.read_bytes()

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode

    mode_raw = mode.encode("ascii", errors="strict")
    if len(mode_raw) > MODE_BYTES:
        raise PayloadFormatError(f"Image mode {mode!r} is too long to store.")

    salt = secrets.token_bytes(SALT_BYTES)
    nonce = secrets.token_bytes(NONCE_BYTES)
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        width,
        height,
        len(mode_raw),
        mode_raw.ljust(MODE_BYTES, b"\0"),
        len(image_bytes) + TAG_SIZE,
        salt,
        nonce,
    )
    cipher = ChaCha20Poly1305(_derive_key(key, salt))
    ciphertext = cipher.encrypt(nonce, image_bytes, header)
    return header + ciphertext


serialize_png = serialize_image


def parse_header(data: bytes) -> PayloadHeader:
    if len(data) < HEADER_SIZE:
        raise PayloadFormatError("Not enough bits were recovered for a payload header.")
    magic, version, width, height, mode_len, mode_raw, payload_length, salt, nonce = (
        HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    )
    if magic != MAGIC:
        raise PayloadFormatError("Bad payload magic number; wrong key or parameters.")
    if version != VERSION:
        raise PayloadFormatError(f"Unsupported payload version {version}.")
    if mode_len > MODE_BYTES:
        raise PayloadFormatError("Invalid payload mode length.")
    try:
        mode = mode_raw[:mode_len].decode("ascii")
    except UnicodeDecodeError as exc:
        raise PayloadFormatError("Invalid payload mode encoding.") from exc
    if width <= 0 or height <= 0 or payload_length <= 0:
        raise PayloadFormatError("Invalid payload dimensions or length.")
    return PayloadHeader(
        width=width,
        height=height,
        mode=mode,
        payload_length=payload_length,
        salt=salt,
        nonce=nonce,
    )


def parse_packet(
    data: bytes, *, key: str, suspect_frames: list[str] | None = None
) -> PayloadPacket:
    header = parse_header(data)
    expected_total = HEADER_SIZE + header.payload_length
    if len(data) < expected_total:
        raise PayloadFormatError(
            f"Payload is truncated: need {expected_total} bytes, recovered {len(data)}."
        )
    ciphertext = data[HEADER_SIZE : HEADER_SIZE + header.payload_length]
    cipher = ChaCha20Poly1305(_derive_key(key, header.salt))
    try:
        image_bytes = cipher.decrypt(header.nonce, ciphertext, data[:HEADER_SIZE])
    except InvalidTag as exc:
        expected_hint = int.from_bytes(ciphertext[-TAG_SIZE : -TAG_SIZE + 4], "big")
        actual_hint = int.from_bytes(
            hashlib.sha256(data[:HEADER_SIZE] + ciphertext).digest()[:4], "big"
        )
        raise AuthenticationError(
            expected_hint, actual_hint, suspect_frames=suspect_frames
        ) from exc
    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise PayloadFormatError(
            "Recovered bytes authenticated but are not a valid image stream."
        ) from exc
    return PayloadPacket(header=header, image_bytes=image_bytes, auth_tag=ciphertext[-TAG_SIZE:])


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8:
        pad = 8 - (bits.size % 8)
        bits = np.pad(bits, (0, pad), constant_values=0)
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()
