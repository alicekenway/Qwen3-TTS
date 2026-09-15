#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec srun \
  -p gpu \
  --gres=gpu:ampere:1 \
  -t 01:00:00 \
  /mnt/users/jinyang_wang/TTS_qwen3TTS/.qwen3_tts_env/bin/python \
  "${SCRIPT_DIR}/generate_tts.py" \
  --input-json /mnt/users/jinyang_wang/TTS_qwen3TTS/temp/input.json \
  --speaker-file /mnt/users/jinyang_wang/TTS_qwen3TTS/temp/speakers.txt \
  --output-dir /mnt/users/jinyang_wang/TTS_qwen3TTS/temp/output \
  --model-path /mnt/users/jinyang_wang/TTS_qwen3TTS/model/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --sample-rate 16000 \
  --device cuda:0 \
  --attn-implementation sdpa \
  "$@"
