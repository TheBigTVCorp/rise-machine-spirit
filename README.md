# Rise Machine Spirit

Rise Machine Spirit is a Python CLI for hiding a key-protected image inside a
lossless WAV or FLAC carrier using DWT-domain QIM steganography. It modifies
detail coefficients at DWT levels 3-5, scatters coefficient locations with a
key-derived PRNG, and encrypts/authenticates the image payload before embedding.

This is a V1 engineering baseline. It includes a compact Bark-band masking
model and analysis output, but it does not claim a formal PEAQ score or a
universal guarantee of inaudibility on every recording. Verify with `--analyze`
and listening tests before using valuable source material.

## Install

```powershell
cd rise-machine-spirit
python -m pip install -e ".[dev]"
```

Python 3.11+ is required. The package uses PyWavelets, numpy, scipy, soundfile,
Pillow, Click, matplotlib, and pytest.

## Usage

```powershell
rms conceal --carrier IN.wav --image BIG.png --output OUT.wav --key "seed phrase" --analyze
rms view --stego OUT.wav --key "seed phrase"
rms reveal-wave --stego OUT.wav --key "seed phrase" --output REVEAL.wav

rms prep-image --input BIG.png --output payload.jpg --carrier IN.wav
rms embed --carrier IN.wav --payload payload.jpg --output OUT.wav --key "seed phrase" --analyze
rms extract --stego OUT.wav --output recovered.jpg --key "seed phrase"
```

`view` authenticates and opens the hidden image only when the key is correct.
`reveal-wave` authenticates the payload and creates a new WAV whose spectrogram
shows the hidden image in Adobe Audition or another spectral audio viewer.
`conceal` is the recommended workflow for large images: it DCT-compresses the
image to the carrier's byte budget, encrypts it, scatters it into WAV/FLAC
coefficients, writes a JSON metadata sidecar, and optionally writes an analysis
diff.

The same key and embedding parameters must be used for extraction. The general
embed/extract defaults are:

```text
frame-size = 4096
overlap = 0.5
wavelet = bior4.4
level = 5
qim-floor = 0.00075
iterations = 2
redundancy = 5
```

The safer `conceal` defaults are `overlap=0`, `qim-floor=0.0012`,
`iterations=3`, and `redundancy=5`. These trade capacity for less coefficient
interference and more reliable viewer recovery.

For very quiet music or 16-bit carriers, lower `--qim-floor` may reduce audible
changes but also reduces extraction margin. For noisy 24-bit WAV/FLAC carriers,
`--qim-floor 0.0012 --iterations 3 --redundancy 5` is a practical starting
point.

## Why WAV or FLAC Only

The payload is carried by exact sample-domain structure after inverse DWT
reconstruction. MP3, AAC, Opus, and similar lossy codecs discard and reshape
audio information according to their own psychoacoustic models. That changes
the QIM coefficient parity and will usually destroy the payload. Use WAV or
FLAC from embedding through extraction.

## Payload Format

The embedded byte stream is:

```text
magic + version + public image metadata + payload length + salt + nonce + AEAD ciphertext
```

The image bytes are encrypted with ChaCha20-Poly1305. The key is derived from
the passphrase using PBKDF2-HMAC-SHA256 with a random salt. Wrong keys fail
authentication and do not produce a partial image.

For large images, use `prep-image` to DCT-compress the source into a JPEG that
fits the carrier capacity. The recovered JPEG payload is byte-for-byte identical
to the compressed payload embedded in the WAV.

`conceal` does this automatically and writes the prepared JPEG next to the WAV
unless `--prepared-output` is supplied.

## Wavelet and Level

The default wavelet is `bior4.4` at DWT level 5. Biorthogonal wavelets give
stable reconstruction and smooth filters, which are useful when small
coefficient changes are overlap-added back into audio. Levels 3-5 target
mid/high detail bands where broadband material tends to mask small changes
better than silence, bass fundamentals, or sparse pure tones. If artifacts show
up on your material, compare `bior4.4`, `bior6.8`, `sym8`, and `db8` with the
same carrier and payload.

## Analysis

`rms embed --analyze` prints:

- full-file SNR between carrier and stego
- segmental SNR
- a spectrogram difference PNG next to the output audio

This is a practical inspection tool, not a standards-compliant PEAQ
implementation.

`rms conceal --analyze` also writes `OUT.wav.metadata.json` with carrier/stego
hashes, sample properties, prepared-payload dimensions, embedding parameters,
packet size, capacity, clipping report, and analysis paths. The passphrase is
not stored in this metadata.

To create a SHA-256 manifest for a paper or demo artifact set:

```powershell
rms hash-artifacts clean.wav stego.wav reveal.wav recovered.jpg --output SHA256SUMS.txt
```

## Tests

```powershell
cd rise-machine-spirit
pytest
```

The tests cover round-trip embed/extract on a 10-second pink-noise carrier,
capacity calculation, authenticated failure handling, MP3 rejection,
metadata-aware lossless loading, image preparation, and the main CLI workflow.

## Example Script

```powershell
cd rise-machine-spirit
python examples\embed_and_extract.py
```

It creates a synthetic 10-second pink-noise WAV, embeds a generated PNG,
extracts it again, and writes a spectrogram diff under `examples\out`.

## Whitepaper Draft

See [docs/whitepaper.md](docs/whitepaper.md) for a practitioner-facing draft
framed around defanged audio-steganography emulation for red and purple teams.
See [REPRODUCE.md](REPRODUCE.md) for manual reproduction commands and artifact
hashing.

## AI-Assisted Development Note

AI assistance was used during prototyping and documentation. The project should
be evaluated by its reviewed source, test coverage, reproducibility steps, and
generated artifacts rather than by authorship guesses or detector scores.
