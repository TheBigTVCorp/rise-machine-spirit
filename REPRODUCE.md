# Reproduce The Defanged Demo

This file documents the manual workflow used for the practitioner demo. Each
stage is an explicit command so a SOC, DFIR, or red-team reader can pause,
inspect artifacts, collect hashes, and try failure cases.

The commands assume PowerShell from the repository root:

```powershell
cd rise-machine-spirit
python -m pip install -e ".[dev]"
rms --help
```

## Inputs

Carrier:

```text
examples\out\music-mp3-demo\Back to Playtime.copy.wav
```

Image:

```text
<path-to-source-image>
```

Key:

```text
crab-wave-secret
```

If the carrier is missing, regenerate or copy a lossless WAV first. Do not use
MP3/AAC/Opus as the embedding carrier; lossy codecs alter the exact samples and
will usually destroy the embedded packet.

## 1. Conceal

```powershell
rms conceal `
  --carrier "examples\out\music-mp3-demo\Back to Playtime.copy.wav" `
  --image "<path-to-source-image>" `
  --output "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.wav" `
  --key "crab-wave-secret" `
  --capacity-ratio 0.45 `
  --analyze
```

Expected artifacts:

- `Back to Playtime.crab-quiet-stego.wav`
- `Back to Playtime.crab-quiet-stego.payload.jpg`
- `Back to Playtime.crab-quiet-stego.wav.metadata.json`
- `Back to Playtime.crab-quiet-stego.wav.spectrogram-diff.png`

Expected observation: the stego WAV should sound like the carrier with, at most,
very faint background hiss depending on playback chain and source material.

## 2. View With Correct Key

```powershell
rms view `
  --stego "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.wav" `
  --key "crab-wave-secret" `
  --output "examples\out\music-mp3-demo\decoded-crab.jpg" `
  --no-open
```

Expected observation: the recovered JPEG is written only after the hidden packet
authenticates.

## 3. View With Wrong Key

```powershell
rms view `
  --stego "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.wav" `
  --key "wrong-key" `
  --output "examples\out\music-mp3-demo\wrong-key.jpg" `
  --no-open
```

Expected observation: the command exits cleanly with an error such as bad magic,
wrong parameters, or payload authentication failure. It should not produce a
usable image.

## 4. Generate Reveal WAV

```powershell
rms reveal-wave `
  --stego "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.wav" `
  --key "crab-wave-secret" `
  --output "examples\out\music-mp3-demo\Back to Playtime.CRAB-REVEAL.wav" `
  --duration 16 `
  --style hybrid
```

Expected observation: open the reveal WAV in Adobe Audition's Spectral Frequency
Display. The reveal WAV is a separate analyst artifact generated after the key
authenticates; the normal stego carrier should not visibly show the image.

## 5. Manual Defanged Activation

```powershell
rms demo-activate `
  --stego "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.wav" `
  --key "crab-wave-secret" `
  --marker "examples\out\music-mp3-demo\CRAB-DEMO-ACTIVATED.txt" `
  --message "CRAB DELIVERY DEMO ACTIVATED"
```

Expected observation: the marker file is a local inert proof that the payload
authenticated. It is not code execution, persistence, injection, networking, or
automatic launch behavior.

## 6. Hash Artifacts

```powershell
rms hash-artifacts `
  "examples\out\music-mp3-demo\Back to Playtime.copy.wav" `
  "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.wav" `
  "examples\out\music-mp3-demo\Back to Playtime.crab-quiet-stego.payload.jpg" `
  "examples\out\music-mp3-demo\Back to Playtime.CRAB-REVEAL.wav" `
  "examples\out\music-mp3-demo\decoded-crab.jpg" `
  "examples\out\music-mp3-demo\CRAB-DEMO-ACTIVATED.txt" `
  --output "examples\out\music-mp3-demo\SHA256SUMS.txt"
```

Expected observation: `SHA256SUMS.txt` records the exact artifact set used in
the paper or demo.

## 7. Test Suite

```powershell
pytest -q
```

Expected observation: all tests pass. The tests cover library round trip,
capacity behavior, authenticated failure, lossy carrier refusal, image
preparation, and the main CLI workflow.
