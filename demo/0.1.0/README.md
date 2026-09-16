# Sequential Base voice-clone demo

`generate_voice_clone.py` generates every target-text/reference-audio pair with
a Qwen3-TTS Base checkpoint. It runs one candidate at a time on one device and
does not import CosyVoice.

## Input

```json
[
  {
    "id": "dialog-001",
    "text": ["Hello.", "Goodbye."],
    "reference_audio_path": ["/path/ref-a.wav", "/path/ref-b.wav"],
    "reference_audio_text": ["Transcript A.", "Transcript B."]
  }
]
```

Reference path and transcript lists must be aligned. Every text is paired with
every reference, so this example produces four independent candidates. Relative
reference paths are resolved against the input JSON directory.

## Run

```bash
python demo/0.1.0/generate_voice_clone.py \
  --model-path /path/Qwen3-TTS-12Hz-1.7B-Base \
  --input-json /path/input.json \
  --output-dir /path/output \
  --language English \
  --sample-rate 16000 \
  --device cuda:0 \
  --attn-implementation sdpa
```

The default seed is zero. Sampling options are inherited from the checkpoint
unless explicitly supplied, for example `--temperature 0.7` or
`--no-do-sample`. Rerunning a compatible partial output resumes by candidate
ID. Use `--overwrite` to deliberately restart all candidates.

Outputs include grouped `generated.json`/`failed.json`, flat TSV files, an
expanded manifest, generation state, and PCM-16 WAV files under `wav/`.
