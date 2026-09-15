#!/usr/bin/env python3
"""Generate a resumable multi-speaker dataset with Qwen3-TTS CustomVoice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Sequence


STATE_VERSION = 1
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAFE_SPEAKER = re.compile(r"[A-Za-z0-9_.-]+")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate every JSON input for every requested CustomVoice speaker."
    )
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--speaker-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=positive_int, default=16000)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help="Transformers attention backend (default: sdpa).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete requested speaker outputs and regenerate them from item 1.",
    )
    return parser.parse_args(argv)


def load_inputs(path: Path) -> list[dict[str, str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Input JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError("Input JSON must be one object or a non-empty list of objects")

    items: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Input item {index} must be a JSON object")
        text = item.get("text")
        prompt = item.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Input item {index} has a missing or empty string 'text'")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Input item {index} has a missing or empty string 'prompt'")
        items.append({"text": text, "prompt": prompt})
    return items


def load_speakers(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"Speaker file does not exist: {path}") from exc

    speakers = [line.strip() for line in lines if line.strip()]
    if not speakers:
        raise ValueError("Speaker file contains no speaker names")

    seen: set[str] = set()
    for speaker in speakers:
        if not SAFE_SPEAKER.fullmatch(speaker) or speaker in {".", ".."}:
            raise ValueError(
                f"Unsafe speaker name {speaker!r}; use only letters, digits, '.', '_' or '-'"
            )
        key = speaker.casefold()
        if key in seen:
            raise ValueError(f"Duplicate speaker name (case-insensitive): {speaker}")
        seen.add(key)
    return speakers


def state_payload(
    items: list[dict[str, str]],
    speaker: str,
    model_path: str,
    language: str,
    sample_rate: int,
    attn_implementation: str,
) -> dict[str, Any]:
    inputs_json = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": STATE_VERSION,
        "input_sha256": hashlib.sha256(inputs_json.encode("utf-8")).hexdigest(),
        "speaker": speaker,
        "model_path": model_path,
        "language": language,
        "sample_rate": sample_rate,
        "attn_implementation": attn_implementation,
    }


def expected_record(index: int, text: str) -> dict[str, str]:
    return {"audiofile_path": f"wav/{index:06d}.wav", "text": text}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    records: list[dict[str, str]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Metadata row at {path}:{line_number} is not an object")
            records.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return records


def inspect_resume(
    speaker_dir: Path,
    expected_state: dict[str, Any],
    items: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not speaker_dir.exists():
        return []
    if not speaker_dir.is_dir() or speaker_dir.is_symlink():
        raise ValueError(f"Speaker output is not a safe directory: {speaker_dir}")

    state_path = speaker_dir / ".generation_state.json"
    if not state_path.exists():
        raise ValueError(
            f"Existing output has no generation state: {speaker_dir}. Use --overwrite to restart."
        )
    if read_json(state_path) != expected_state:
        raise ValueError(
            f"Existing output does not match the current inputs/settings: {speaker_dir}. "
            "Use --overwrite to restart."
        )

    records = read_metadata(speaker_dir / "metadata.jsonl")
    if len(records) > len(items):
        raise ValueError(f"Metadata has more rows than the input: {speaker_dir / 'metadata.jsonl'}")

    completed: list[dict[str, str]] = []
    missing_wav_found = False
    for offset, record in enumerate(records):
        index = offset + 1
        expected = expected_record(index, items[offset]["text"])
        if record != expected:
            raise ValueError(
                f"Metadata row {index} does not match the current input in "
                f"{speaker_dir / 'metadata.jsonl'}. Use --overwrite to restart."
            )
        wav_path = speaker_dir / expected["audiofile_path"]
        if not wav_path.is_file():
            missing_wav_found = True
        elif missing_wav_found:
            raise ValueError(
                f"Metadata/WAV completion is not a contiguous prefix in {speaker_dir}. "
                "Use --overwrite to restart."
            )
        else:
            completed.append(expected)
    return completed


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_metadata(path: Path, records: list[dict[str, str]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    atomic_write_text(path, content)


def default_model_loader(model_path: str, device: str, attn_implementation: str) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.float32 if device.casefold().startswith("cpu") else torch.bfloat16
    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def default_audio_writer(wav: Any, source_rate: int, target: Path, target_rate: int) -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    audio = np.asarray(wav, dtype=np.float32)
    if source_rate != target_rate:
        audio = librosa.resample(y=audio, orig_sr=source_rate, target_sr=target_rate)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".wav", dir=target.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        sf.write(temporary_path, audio, target_rate, subtype="PCM_16")
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_supported_speakers(model: Any, speakers: list[str]) -> None:
    supported = model.get_supported_speakers()
    if supported is None:
        return
    supported_keys = {str(name).casefold() for name in supported}
    invalid = [speaker for speaker in speakers if speaker.casefold() not in supported_keys]
    if invalid:
        raise ValueError(
            f"Unsupported speakers: {invalid}. Supported: {', '.join(sorted(supported))}"
        )


def generate_dataset(
    args: argparse.Namespace,
    model_loader: Callable[[str, str], Any] = default_model_loader,
    audio_writer: Callable[[Any, int, Path, int], None] = default_audio_writer,
) -> None:
    items = load_inputs(args.input_json)
    speakers = load_speakers(args.speaker_file)
    output_dir = args.output_dir.resolve()

    states = {
        speaker: state_payload(
            items,
            speaker,
            args.model_path,
            args.language,
            args.sample_rate,
            args.attn_implementation,
        )
        for speaker in speakers
    }
    completed_by_speaker: dict[str, list[dict[str, str]]] = {}
    if args.overwrite:
        completed_by_speaker = {speaker: [] for speaker in speakers}
    else:
        for speaker in speakers:
            completed_by_speaker[speaker] = inspect_resume(
                output_dir / speaker, states[speaker], items
            )

    total = len(items) * len(speakers)
    skipped = sum(len(records) for records in completed_by_speaker.values())
    print(f"Input rows: {len(items)}")
    print(f"Speakers: {len(speakers)} ({', '.join(speakers)})")

    if skipped == total:
        print("Generated utterances: 0")
        print(f"Skipped utterances: {skipped}")
        print("Failed rows: 0")
        return

    model = model_loader(args.model_path, args.device, args.attn_implementation)
    validate_supported_speakers(model, speakers)

    if args.overwrite:
        for speaker in speakers:
            speaker_dir = output_dir / speaker
            if speaker_dir.exists():
                if not speaker_dir.is_dir() or speaker_dir.is_symlink():
                    raise ValueError(f"Refusing to overwrite unsafe path: {speaker_dir}")
                shutil.rmtree(speaker_dir)

    generated = 0
    failed = 0
    try:
        for speaker in speakers:
            speaker_dir = output_dir / speaker
            wav_dir = speaker_dir / "wav"
            wav_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                speaker_dir / ".generation_state.json",
                json.dumps(states[speaker], ensure_ascii=False, indent=2) + "\n",
            )

            completed = completed_by_speaker[speaker]
            write_metadata(speaker_dir / "metadata.jsonl", completed)
            for offset in range(len(completed), len(items)):
                index = offset + 1
                item = items[offset]
                wavs, source_rate = model.generate_custom_voice(
                    text=item["text"],
                    speaker=speaker,
                    language=args.language,
                    instruct=item["prompt"],
                )
                if len(wavs) != 1:
                    raise RuntimeError(
                        f"Expected one waveform for {speaker} item {index}, got {len(wavs)}"
                    )
                record = expected_record(index, item["text"])
                audio_writer(
                    wavs[0], source_rate, speaker_dir / record["audiofile_path"], args.sample_rate
                )
                completed.append(record)
                write_metadata(speaker_dir / "metadata.jsonl", completed)
                generated += 1
                print(f"Generated {speaker} item {index}/{len(items)}")
    except BaseException:
        failed = 1
        raise
    finally:
        print(f"Generated utterances: {generated}")
        print(f"Skipped utterances: {skipped}")
        print(f"Failed rows: {failed}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        generate_dataset(parse_args(argv))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
