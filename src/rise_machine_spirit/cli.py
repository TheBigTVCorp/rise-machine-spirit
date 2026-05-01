from __future__ import annotations

import hashlib
import json
import string
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import click
from PIL import Image

from .analysis import save_spectrogram_diff, segmental_snr_db, snr_db
from .audio import read_lossless_audio
from .exceptions import RiseMachineSpiritError
from .imageprep import prepare_jpeg_to_budget
from .spectro_reveal import image_to_spectrogram_wave
from .stego import StegoParams, embed_png, estimate_capacity, extract_image, extract_png


def _params(
    frame_size: int,
    overlap: float,
    wavelet: str,
    level: int,
    qim_floor: float,
    iterations: int,
    redundancy: int,
) -> StegoParams:
    return StegoParams(
        frame_size=frame_size,
        overlap=overlap,
        wavelet=wavelet,
        level=level,
        qim_floor=qim_floor,
        iterations=iterations,
        redundancy=redundancy,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Embed, extract, and view key-protected image payloads in lossless audio."""


COMMON_OPTIONS = [
    click.option("--frame-size", default=4096, show_default=True, type=int),
    click.option("--overlap", default=0.5, show_default=True, type=float),
    click.option("--wavelet", default="bior4.4", show_default=True),
    click.option("--level", "dwt_level", default=5, show_default=True, type=int),
    click.option("--qim-floor", default=7.5e-4, show_default=True, type=float),
]


def _apply_common_options(function):
    for option in reversed(COMMON_OPTIONS):
        function = option(function)
    return function


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_metadata(path: Path) -> dict[str, object]:
    audio, meta = read_lossless_audio(path)
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "samplerate": meta.samplerate,
        "channels": meta.channels,
        "format": meta.format,
        "subtype": meta.subtype,
        "samples": int(audio.shape[0]),
        "duration_seconds": audio.shape[0] / meta.samplerate,
    }


def _params_metadata(params: StegoParams) -> dict[str, object]:
    return {
        "frame_size": params.frame_size,
        "overlap": params.overlap,
        "wavelet": params.wavelet,
        "level": params.level,
        "qim_floor": params.qim_floor,
        "iterations": params.iterations,
        "redundancy": params.redundancy,
    }


def _print_clip_warning(result) -> None:
    if result.clipped_samples:
        click.echo(
            "Warning: output clipping occurred: "
            f"{result.clipped_samples} sample(s), peak before clip {result.peak_before_clip:.6f}."
        )


def _print_analysis(carrier: Path, output: Path) -> dict[str, object]:
    original, meta = read_lossless_audio(carrier)
    stego, _ = read_lossless_audio(output)
    snr = snr_db(original, stego)
    segmental = segmental_snr_db(original, stego)
    click.echo(f"SNR: {snr:.2f} dB")
    click.echo(f"Segmental SNR: {segmental:.2f} dB")
    diff_path = output.with_suffix(output.suffix + ".spectrogram-diff.png")
    save_spectrogram_diff(original, stego, meta.samplerate, diff_path)
    click.echo(f"Spectrogram diff: {diff_path}")
    return {
        "snr_db": snr,
        "segmental_snr_db": segmental,
        "spectrogram_diff": str(diff_path),
        "spectrogram_diff_sha256": _file_sha256(diff_path),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in string.hexdigits for char in normalized):
        raise click.ClickException("--expect-sha256 must be a 64-character SHA-256 hex digest.")
    return normalized


def _show_message_box(title: str, message: str) -> None:
    if sys.platform != "win32":
        raise click.ClickException("--popup is only supported on Windows.")
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, message, title, 0x40)


@main.command()
@click.option(
    "--carrier", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--payload", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--key", required=True)
@click.option("--analyze", is_flag=True, help="Print SNR metrics and write a spectrogram diff PNG.")
@click.option("--iterations", default=2, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
@_apply_common_options
def embed(
    carrier: Path,
    payload: Path,
    output: Path,
    key: str,
    analyze: bool,
    iterations: int,
    redundancy: int,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
) -> None:
    """Embed an image payload into a WAV or FLAC carrier."""

    params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, iterations, redundancy)
    try:
        result = embed_png(carrier, payload, output, key=key, params=params)
        click.echo(
            f"Embedded {result.payload_bytes} bytes; capacity {result.capacity_bytes} bytes "
            f"at {result.sample_rate} Hz/{result.channels} channel(s)."
        )
        _print_clip_warning(result)
        if analyze:
            _print_analysis(carrier, output)
    except RiseMachineSpiritError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option(
    "--stego", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--key", required=True)
@click.option("--iterations", default=2, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
@_apply_common_options
def extract(
    stego: Path,
    output: Path,
    key: str,
    iterations: int,
    redundancy: int,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
) -> None:
    """Extract an image payload from a WAV or FLAC stego file."""

    params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, iterations, redundancy)
    try:
        extract_png(stego, output, key=key, params=params)
        click.echo(f"Recovered image: {output}")
    except RiseMachineSpiritError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option(
    "--stego", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--key", required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--open/--no-open", "open_image", default=True, show_default=True)
@click.option("--frame-size", default=4096, show_default=True, type=int)
@click.option("--overlap", default=0.0, show_default=True, type=float)
@click.option("--wavelet", default="bior4.4", show_default=True)
@click.option("--level", "dwt_level", default=5, show_default=True, type=int)
@click.option("--qim-floor", default=0.0012, show_default=True, type=float)
@click.option("--iterations", default=3, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
def view(
    stego: Path,
    key: str,
    output: Path | None,
    open_image: bool,
    iterations: int,
    redundancy: int,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
) -> None:
    """Recover and open the hidden image only when the key authenticates."""

    params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, iterations, redundancy)
    try:
        packet = extract_image(stego, key=key, params=params)
        with Image.open(BytesIO(packet.image_bytes)) as image:
            image_format = (image.format or "PNG").lower()
        suffix = ".jpg" if image_format == "jpeg" else f".{image_format}"
        image_path = output
        if image_path is None:
            with tempfile.NamedTemporaryFile(
                prefix="rms-view-", suffix=suffix, delete=False
            ) as temp:
                temp.write(packet.image_bytes)
                image_path = Path(temp.name)
        else:
            image_path.write_bytes(packet.image_bytes)
        click.echo(f"Authenticated hidden image: {image_path}")
        if open_image:
            click.launch(str(image_path))
    except RiseMachineSpiritError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("reveal-wave")
@click.option(
    "--stego", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--key", required=True)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--duration", default=12.0, show_default=True, type=float)
@click.option("--min-freq", default=1000.0, show_default=True, type=float)
@click.option("--max-freq", default=20000.0, show_default=True, type=float)
@click.option(
    "--style", type=click.Choice(["image", "hybrid", "edges"]), default="hybrid", show_default=True
)
@click.option("--frame-size", default=4096, show_default=True, type=int)
@click.option("--overlap", default=0.0, show_default=True, type=float)
@click.option("--wavelet", default="bior4.4", show_default=True)
@click.option("--level", "dwt_level", default=5, show_default=True, type=int)
@click.option("--qim-floor", default=0.0012, show_default=True, type=float)
@click.option("--iterations", default=3, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
def reveal_wave(
    stego: Path,
    key: str,
    output: Path,
    duration: float,
    min_freq: float,
    max_freq: float,
    style: str,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
    iterations: int,
    redundancy: int,
) -> None:
    """Create a reveal WAV whose spectrogram shows the hidden image."""

    params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, iterations, redundancy)
    try:
        packet = extract_image(stego, key=key, params=params)
        image_to_spectrogram_wave(
            packet.image_bytes,
            output,
            duration=duration,
            min_freq=min_freq,
            max_freq=max_freq,
            style=style,
        )
        click.echo(f"Reveal WAV: {output}")
        click.echo("Open it in Adobe Audition spectral frequency display to see the image.")
    except (RiseMachineSpiritError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("demo-activate")
@click.option(
    "--stego", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--key", required=True)
@click.option("--marker", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--message", default="DEMO PAYLOAD ACTIVATED", show_default=True)
@click.option("--expect-sha256", help="Require the recovered hidden gate to match this SHA-256.")
@click.option("--reveal-output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--proof-output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--popup/--no-popup", default=False, show_default=True)
@click.option("--popup-title", default="Project Sgt-Sigar", show_default=True)
@click.option("--frame-size", default=4096, show_default=True, type=int)
@click.option("--overlap", default=0.0, show_default=True, type=float)
@click.option("--wavelet", default="bior4.4", show_default=True)
@click.option("--level", "dwt_level", default=5, show_default=True, type=int)
@click.option("--qim-floor", default=0.0012, show_default=True, type=float)
@click.option("--iterations", default=3, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
def demo_activate(
    stego: Path,
    key: str,
    marker: Path,
    message: str,
    expect_sha256: str | None,
    reveal_output: Path | None,
    proof_output: Path | None,
    popup: bool,
    popup_title: str,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
    iterations: int,
    redundancy: int,
) -> None:
    """Run a harmless explicit activation after hidden payload authentication."""

    params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, iterations, redundancy)
    try:
        packet = extract_image(stego, key=key, params=params)
        digest = hashlib.sha256(packet.image_bytes).hexdigest()
        expected_digest = _normalize_sha256(expect_sha256) if expect_sha256 else None
        if expected_digest and digest != expected_digest:
            raise click.ClickException(
                f"Recovered hidden gate SHA-256 {digest} does not match expected {expected_digest}."
            )
        reveal_hash = None
        if reveal_output is not None:
            image_to_spectrogram_wave(
                packet.image_bytes, reveal_output, duration=16, style="hybrid"
            )
            reveal_hash = _file_sha256(reveal_output)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "\n".join(
                [
                    message,
                    f"stego={stego}",
                    f"hidden_payload_sha256={digest}",
                    *(
                        [f"expected_payload_sha256={expected_digest}", "gate_verified=true"]
                        if expected_digest
                        else []
                    ),
                    f"hidden_payload_bytes={len(packet.image_bytes)}",
                    *(
                        [f"reveal={reveal_output}", f"reveal_sha256={reveal_hash}"]
                        if reveal_output
                        else []
                    ),
                    f"popup={'shown' if popup else 'not_requested'}",
                    "note=this is an inert local demo action, not code execution",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if popup:
            _show_message_box(popup_title, message)
        proof = {
            "command": "demo-activate",
            "stego": str(stego),
            "hidden_payload_sha256": digest,
            "expected_payload_sha256": expected_digest,
            "gate_verified": expected_digest is None or digest == expected_digest,
            "hidden_payload_bytes": len(packet.image_bytes),
            "marker": str(marker),
            "marker_sha256": _file_sha256(marker),
            "popup": popup,
            "popup_title": popup_title if popup else None,
            "reveal": str(reveal_output) if reveal_output else None,
            "reveal_sha256": reveal_hash,
            "normal_image_written": False,
            "safety": "Recovered bytes are authenticated as gate material and are not executed as code.",
        }
        if proof_output is not None:
            _write_json(proof_output, proof)
            click.echo(f"Proof JSON: {proof_output}")
        if reveal_output is not None:
            click.echo(f"Reveal WAV: {reveal_output}")
        click.echo(f"Recovered hidden gate SHA-256: {digest}")
        if expected_digest:
            click.echo("Gate verification: passed")
        if popup:
            click.echo("Displayed inert popup after gate verification.")
        click.echo(f"Authenticated hidden payload and wrote marker: {marker}")
    except RiseMachineSpiritError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("hash-artifacts")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
def hash_artifacts(paths: tuple[Path, ...], output: Path | None) -> None:
    """Write a SHA-256 manifest for generated demo artifacts."""

    if not paths:
        raise click.ClickException("Provide at least one artifact path to hash.")
    lines = [f"{_file_sha256(path)}  {path}" for path in paths]
    manifest = "\n".join(lines) + "\n"
    if output is None:
        click.echo(manifest, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest, encoding="utf-8")
    click.echo(f"SHA-256 manifest: {output}")


@main.command()
@click.option(
    "--carrier", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--image",
    "image_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--key", required=True)
@click.option("--prepared-output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--metadata-output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--quality-min", default=30, show_default=True, type=int)
@click.option("--quality-max", default=92, show_default=True, type=int)
@click.option("--capacity-ratio", default=0.7, show_default=True, type=float)
@click.option("--analyze", is_flag=True, help="Print SNR metrics and write a spectrogram diff PNG.")
@click.option("--frame-size", default=4096, show_default=True, type=int)
@click.option("--overlap", default=0.0, show_default=True, type=float)
@click.option("--wavelet", default="bior4.4", show_default=True)
@click.option("--level", "dwt_level", default=5, show_default=True, type=int)
@click.option("--qim-floor", default=0.0012, show_default=True, type=float)
@click.option("--iterations", default=3, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
def conceal(
    carrier: Path,
    image_path: Path,
    output: Path,
    key: str,
    prepared_output: Path | None,
    metadata_output: Path | None,
    quality_min: int,
    quality_max: int,
    capacity_ratio: float,
    analyze: bool,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
    iterations: int,
    redundancy: int,
) -> None:
    """Compress, encrypt, and hide an image in a normal-looking WAV/FLAC."""

    params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, iterations, redundancy)
    try:
        capacity = estimate_capacity(carrier, params=params)
        # Leave room for the encrypted packet header, AEAD tag, and small future metadata.
        if not 0.1 <= capacity_ratio <= 0.98:
            raise click.ClickException("--capacity-ratio must be between 0.1 and 0.98.")
        budget = max(0, int((capacity - 256) * capacity_ratio))
        if prepared_output is None:
            prepared_output = output.with_suffix(".payload.jpg")
        prepared = prepare_jpeg_to_budget(
            image_path,
            prepared_output,
            budget,
            quality_min=quality_min,
            quality_max=quality_max,
        )
        click.echo(
            f"DCT payload: {prepared.path} q{prepared.quality}, "
            f"{prepared.size_bytes}/{prepared.budget_bytes} bytes, {prepared.dimensions[0]}x{prepared.dimensions[1]}."
        )
        result = embed_png(carrier, prepared.path, output, key=key, params=params)
        click.echo(
            f"Concealed encrypted image in {output}; packet {result.payload_bytes} bytes, "
            f"capacity {result.capacity_bytes} bytes."
        )
        _print_clip_warning(result)
        analysis_metrics = None
        if analyze:
            analysis_metrics = _print_analysis(carrier, output)
        metadata_path = metadata_output or output.with_suffix(output.suffix + ".metadata.json")
        _write_json(
            metadata_path,
            {
                "command": "conceal",
                "carrier": _audio_metadata(carrier),
                "stego": _audio_metadata(output),
                "source_image": {
                    "path": str(image_path),
                    "sha256": _file_sha256(image_path),
                },
                "prepared_payload": {
                    "path": str(prepared.path),
                    "sha256": _file_sha256(prepared.path),
                    "size_bytes": prepared.size_bytes,
                    "budget_bytes": prepared.budget_bytes,
                    "jpeg_quality": prepared.quality,
                    "dimensions": list(prepared.dimensions),
                },
                "embedding": {
                    "params": _params_metadata(params),
                    "capacity_ratio": capacity_ratio,
                    "capacity_bytes": result.capacity_bytes,
                    "packet_bytes": result.payload_bytes,
                    "clipped_samples": result.clipped_samples,
                    "peak_before_clip": result.peak_before_clip,
                },
                "analysis": analysis_metrics,
                "notes": [
                    "The passphrase is intentionally not written to metadata.",
                    "WAV/FLAC sample properties are preserved; ancillary RIFF chunks are not guaranteed.",
                ],
            },
        )
        click.echo(f"Metadata: {metadata_path}")
    except (RiseMachineSpiritError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("prep-image")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--carrier", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-bytes", type=int)
@click.option("--quality-min", default=30, show_default=True, type=int)
@click.option("--quality-max", default=92, show_default=True, type=int)
@click.option("--redundancy", default=5, show_default=True, type=int)
@_apply_common_options
def prep_image(
    input_path: Path,
    output: Path,
    carrier: Path | None,
    target_bytes: int | None,
    quality_min: int,
    quality_max: int,
    redundancy: int,
    frame_size: int,
    overlap: float,
    wavelet: str,
    dwt_level: int,
    qim_floor: float,
) -> None:
    """DCT-compress an image to fit a target byte budget or carrier capacity."""

    if carrier is None and target_bytes is None:
        raise click.ClickException("Provide --carrier or --target-bytes.")
    if carrier is not None:
        params = _params(frame_size, overlap, wavelet, dwt_level, qim_floor, 1, redundancy)
        budget = estimate_capacity(carrier, params=params) - 128
    else:
        budget = int(target_bytes or 0)
    if budget <= 0:
        raise click.ClickException("Target byte budget is too small.")

    try:
        prepared = prepare_jpeg_to_budget(
            input_path,
            output,
            budget,
            quality_min=quality_min,
            quality_max=quality_max,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Wrote {prepared.path} at JPEG quality {prepared.quality}: "
        f"{prepared.size_bytes} bytes for budget {prepared.budget_bytes} bytes, "
        f"{prepared.dimensions[0]}x{prepared.dimensions[1]}."
    )


if __name__ == "__main__":
    main()
