from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from PIL import Image

from rise_machine_spirit.audio import read_lossless_audio
from rise_machine_spirit.cli import main
from rise_machine_spirit.exceptions import AuthenticationError, UnsupportedCarrierError
from rise_machine_spirit.imageprep import prepare_jpeg_to_budget
from rise_machine_spirit.payload import HEADER_SIZE, parse_packet, serialize_image
from rise_machine_spirit.stego import StegoParams, embed_png, estimate_capacity, extract_png


def _pink_noise(seconds: float = 10.0, samplerate: int = 48_000, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(1234)
    samples = int(seconds * samplerate)
    freqs = np.fft.rfftfreq(samples, d=1.0 / samplerate)
    shaped = []
    for _channel in range(channels):
        white = rng.normal(0, 1, samples)
        spectrum = np.fft.rfft(white)
        scale = np.ones_like(freqs)
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
        pink = np.fft.irfft(spectrum * scale, n=samples)
        pink = pink / np.max(np.abs(pink)) * 0.35
        shaped.append(pink)
    return np.stack(shaped, axis=1).astype(np.float64)


def _write_png(path: Path) -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    for y in range(8):
        for x in range(8):
            image.putpixel((x, y), (x * 31, y * 31, 180, 255))
    image.save(path, format="PNG", optimize=True)


def test_round_trip_embed_extract_10_second_carrier(tmp_path: Path) -> None:
    carrier = tmp_path / "carrier.wav"
    payload = tmp_path / "payload.png"
    stego = tmp_path / "stego.wav"
    recovered = tmp_path / "recovered.png"
    sf.write(carrier, _pink_noise(), 48_000, subtype="PCM_24")
    _write_png(payload)

    params = StegoParams(qim_floor=0.0012, iterations=3)
    result = embed_png(carrier, payload, stego, key="test-seed", params=params)
    extract_png(stego, recovered, key="test-seed", params=params)

    assert result.capacity_bytes > result.payload_bytes
    assert recovered.read_bytes() == payload.read_bytes()


def test_capacity_calculation_is_deterministic_and_scales_with_channels() -> None:
    mono = np.zeros((48_000, 1), dtype=np.float64)
    stereo = np.zeros((48_000, 2), dtype=np.float64)
    params = StegoParams()

    mono_capacity = estimate_capacity(mono, params=params)
    stereo_capacity = estimate_capacity(stereo, params=params)

    assert stereo_capacity > 1000
    assert abs(stereo_capacity - mono_capacity * 2) <= 1


def test_authentication_validation_on_tampered_payload_packet(tmp_path: Path) -> None:
    payload = tmp_path / "payload.png"
    _write_png(payload)
    packet = bytearray(serialize_image(payload, key="packet-key"))
    packet[HEADER_SIZE + 3] ^= 0x40

    with pytest.raises(AuthenticationError):
        parse_packet(bytes(packet), key="packet-key", suspect_frames=["frame=3/ch=0/cD4"])


def test_wrong_payload_key_does_not_decrypt(tmp_path: Path) -> None:
    payload = tmp_path / "payload.png"
    _write_png(payload)
    packet = serialize_image(payload, key="correct-key")

    with pytest.raises(AuthenticationError):
        parse_packet(packet, key="wrong-key")


def test_rejects_mp3_input_before_decode(tmp_path: Path) -> None:
    carrier = tmp_path / "carrier.mp3"
    payload = tmp_path / "payload.png"
    output = tmp_path / "out.wav"
    carrier.write_bytes(b"not really an mp3")
    _write_png(payload)

    with pytest.raises(UnsupportedCarrierError):
        embed_png(carrier, payload, output, key="seed")


def test_read_preserves_lossless_metadata(tmp_path: Path) -> None:
    carrier = tmp_path / "carrier.wav"
    sf.write(carrier, _pink_noise(seconds=0.25), 48_000, subtype="PCM_24")

    audio, meta = read_lossless_audio(carrier)

    assert audio.dtype == np.float64
    assert meta.samplerate == 48_000
    assert meta.subtype == "PCM_24"


def test_prepare_jpeg_to_budget_preserves_dimensions_and_fits(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "prepared.jpg"
    image = Image.new("RGB", (64, 48), (80, 120, 180))
    image.save(source, "PNG")

    prepared = prepare_jpeg_to_budget(source, output, 5000, quality_min=30, quality_max=90)

    assert prepared.size_bytes <= 5000
    assert prepared.dimensions == (64, 48)
    with Image.open(output) as recovered:
        assert recovered.size == (64, 48)
        assert recovered.format == "JPEG"


def test_prepare_jpeg_to_budget_downscales_when_needed(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "prepared.jpg"
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    Image.fromarray(noisy, "RGB").save(source, "PNG")

    prepared = prepare_jpeg_to_budget(source, output, 20_000, quality_min=25, quality_max=90)

    assert prepared.size_bytes <= 20_000
    assert prepared.dimensions[0] < 512
    assert prepared.dimensions[1] < 512


def test_cli_manual_workflow(tmp_path: Path) -> None:
    carrier = tmp_path / "carrier.wav"
    image = tmp_path / "source.png"
    stego = tmp_path / "stego.wav"
    recovered = tmp_path / "recovered.jpg"
    reveal = tmp_path / "reveal.wav"
    proof_reveal = tmp_path / "proof-reveal.wav"
    marker = tmp_path / "marker.txt"
    proof = tmp_path / "proof.json"
    manifest = tmp_path / "SHA256SUMS.txt"
    sf.write(carrier, _pink_noise(seconds=10.0), 48_000, subtype="PCM_24")
    _write_png(image)

    runner = CliRunner()
    conceal_result = runner.invoke(
        main,
        [
            "conceal",
            "--carrier",
            str(carrier),
            "--image",
            str(image),
            "--output",
            str(stego),
            "--key",
            "cli-key",
            "--capacity-ratio",
            "0.45",
        ],
    )
    assert conceal_result.exit_code == 0, conceal_result.output
    assert stego.exists()
    assert stego.with_suffix(".payload.jpg").exists()
    assert Path(str(stego) + ".metadata.json").exists()

    view_result = runner.invoke(
        main,
        [
            "view",
            "--stego",
            str(stego),
            "--key",
            "cli-key",
            "--output",
            str(recovered),
            "--no-open",
        ],
    )
    assert view_result.exit_code == 0, view_result.output
    assert recovered.exists()

    wrong_key_result = runner.invoke(
        main,
        [
            "view",
            "--stego",
            str(stego),
            "--key",
            "wrong-key",
            "--output",
            str(tmp_path / "wrong-key.jpg"),
            "--no-open",
        ],
    )
    assert wrong_key_result.exit_code != 0

    reveal_result = runner.invoke(
        main,
        [
            "reveal-wave",
            "--stego",
            str(stego),
            "--key",
            "cli-key",
            "--output",
            str(reveal),
            "--duration",
            "0.5",
        ],
    )
    assert reveal_result.exit_code == 0, reveal_result.output
    assert reveal.exists()

    expected_digest = hashlib.sha256(recovered.read_bytes()).hexdigest()
    activate_result = runner.invoke(
        main,
        [
            "demo-activate",
            "--stego",
            str(stego),
            "--key",
            "cli-key",
            "--marker",
            str(marker),
            "--message",
            "CLI DEMO ACTIVATED",
            "--expect-sha256",
            expected_digest,
            "--reveal-output",
            str(proof_reveal),
            "--proof-output",
            str(proof),
        ],
    )
    assert activate_result.exit_code == 0, activate_result.output
    assert "CLI DEMO ACTIVATED" in marker.read_text(encoding="utf-8")
    assert "gate_verified=true" in marker.read_text(encoding="utf-8")
    assert proof.exists()
    assert proof_reveal.exists()

    rejected_marker = tmp_path / "rejected-marker.txt"
    rejected_result = runner.invoke(
        main,
        [
            "demo-activate",
            "--stego",
            str(stego),
            "--key",
            "cli-key",
            "--marker",
            str(rejected_marker),
            "--expect-sha256",
            "0" * 64,
        ],
    )
    assert rejected_result.exit_code != 0
    assert not rejected_marker.exists()

    hash_result = runner.invoke(
        main,
        [
            "hash-artifacts",
            str(carrier),
            str(stego),
            str(recovered),
            str(reveal),
            str(marker),
            "--output",
            str(manifest),
        ],
    )
    assert hash_result.exit_code == 0, hash_result.output
    assert manifest.read_text(encoding="utf-8").count("\n") == 5
