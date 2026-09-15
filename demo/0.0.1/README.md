# Multi-speaker TTS demo

This demo generates every JSON input for every named Qwen3-TTS CustomVoice
speaker. It uses the 1.7B model because the 0.6B CustomVoice model does not
apply instruction prompts.

Install the project first (for example, `pip install -e .`), then run from the
repository root:

```bash
python demo/0.0.1/generate_tts.py \
  --input-json demo/0.0.1/example_input.json \
  --speaker-file demo/0.0.1/example_speakers.txt \
  --output-dir /tmp/qwen3_tts_output
```

The JSON file may contain one object or a list. Every object must have
non-empty string `text` and `prompt` fields:

```json
{"text": "Hello", "prompt": "speaking quickly in a joyful tone"}
```

The speaker file contains one speaker per line. The released model supports
`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`,
and `Sohee`. Names are checked case-insensitively by the model. `John` is not a
released built-in speaker and will be rejected.

Useful options:

```text
--sample-rate 16000
--model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
--device cuda:0
--language Auto
--overwrite
```

Each speaker gets a separate directory:

```text
output/Vivian/
├── metadata.jsonl
├── .generation_state.json
└── wav/
    └── 000001.wav
```

Each metadata row has only the relative WAV path and source text:

```json
{"audiofile_path": "wav/000001.wav", "text": "Hello"}
```

Rerunning the same command resumes from the first unfinished item. A completed
run is skipped without loading the model. If the input, prompt, model, language,
speaker, or sample rate changes, use `--overwrite` to deliberately regenerate
the requested speaker directories from the beginning.
