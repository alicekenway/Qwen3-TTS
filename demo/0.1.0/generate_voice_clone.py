#!/usr/bin/env python3
"""Sequential, resumable Qwen3-TTS Base voice cloning from JSON."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Callable, Sequence


STATE_VERSION = 1
GENERATED_FIELDS = ["speechpath", "text", "id"]
FAILED_FIELDS = [
    "row_id",
    "id",
    "text",
    "ref_audio",
    "ref_audio_text",
    "error",
]


@dataclass(frozen=True)
class Conversation:
    index: int
    input_id: str
    texts: list[str]
    reference_audio_paths: list[str]
    reference_audio_texts: list[str]


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    candidate_id: str
    input_index: int
    input_id: str
    text_index: int
    reference_index: int
    text: str
    reference_audio_path: str
    reference_audio_text: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3-TTS Base voice-clone candidates from JSON."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--sample-rate", type=positive_int, default=16000)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--on-error", choices=("skip", "raise"), default="skip"
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--top-k", type=positive_int, default=None)
    parser.add_argument("--top-p", type=probability, default=None)
    parser.add_argument("--temperature", type=positive_float, default=None)
    parser.add_argument("--repetition-penalty", type=positive_float, default=None)
    parser.add_argument(
        "--subtalker-do-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--subtalker-top-k", type=positive_int, default=None)
    parser.add_argument("--subtalker-top-p", type=probability, default=None)
    parser.add_argument(
        "--subtalker-temperature", type=positive_float, default=None
    )
    parser.add_argument("--max-new-tokens", type=positive_int, default=None)
    return parser.parse_args(argv)


def require_string_list(value: Any, record_index: int, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Record {record_index} field '{field}' must be a list")
    output: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Record {record_index} field '{field}' item {item_index} "
                "must be a non-empty string"
            )
        output.append(item.strip())
    if not output:
        raise ValueError(f"Record {record_index} field '{field}' cannot be empty")
    return output


def load_conversations(input_json: Path) -> list[Conversation]:
    try:
        raw = json.loads(input_json.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Input JSON does not exist: {input_json}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_json}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Input JSON must be a non-empty top-level list")

    base_dir = input_json.resolve().parent
    seen_ids: set[str] = set()
    conversations: list[Conversation] = []
    for record_index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"Record {record_index} must be an object")
        input_id = record.get("id")
        if not isinstance(input_id, str) or not input_id.strip():
            raise ValueError(f"Record {record_index} has a missing or empty string 'id'")
        input_id = input_id.strip()
        if input_id in seen_ids:
            raise ValueError(f"Duplicate input id: {input_id}")
        seen_ids.add(input_id)

        texts = require_string_list(record.get("text"), record_index, "text")
        ref_paths = require_string_list(
            record.get("reference_audio_path"), record_index, "reference_audio_path"
        )
        ref_texts = require_string_list(
            record.get("reference_audio_text"), record_index, "reference_audio_text"
        )
        if len(ref_paths) != len(ref_texts):
            raise ValueError(
                f"Record {record_index} reference_audio_path and "
                "reference_audio_text must have equal lengths"
            )

        resolved_paths: list[str] = []
        for reference_index, raw_path in enumerate(ref_paths):
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError(
                    f"Record {record_index} reference {reference_index} does not exist: {path}"
                )
            resolved_paths.append(str(path))

        conversations.append(
            Conversation(
                index=record_index,
                input_id=input_id,
                texts=texts,
                reference_audio_paths=resolved_paths,
                reference_audio_texts=ref_texts,
            )
        )
    return conversations


def candidate_id(input_index: int, text_index: int, reference_index: int) -> str:
    return (
        f"rec{input_index:06d}_text{text_index:06d}_ref{reference_index:06d}"
    )


def expand_candidates(conversations: Sequence[Conversation]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for conversation in conversations:
        for text_index, text in enumerate(conversation.texts):
            for reference_index, ref_path in enumerate(
                conversation.reference_audio_paths
            ):
                candidates.append(
                    Candidate(
                        ordinal=len(candidates),
                        candidate_id=candidate_id(
                            conversation.index, text_index, reference_index
                        ),
                        input_index=conversation.index,
                        input_id=conversation.input_id,
                        text_index=text_index,
                        reference_index=reference_index,
                        text=text,
                        reference_audio_path=ref_path,
                        reference_audio_text=conversation.reference_audio_texts[
                            reference_index
                        ],
                    )
                )
    return candidates


def generation_options(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "subtalker_do_sample",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
        "max_new_tokens",
    )
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def state_payload(args: argparse.Namespace, input_bytes: bytes) -> dict[str, Any]:
    return {
        "schema_version": STATE_VERSION,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "language": args.language,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "generation_options": generation_options(args),
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def tsv_content(fieldnames: list[str], rows: Sequence[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_tsv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Unexpected TSV header in {path}: {reader.fieldnames}; "
                f"expected {expected_fields}"
            )
        return [dict(row) for row in reader]


def grouped_generated_json(
    conversations: Sequence[Conversation], successful_paths: dict[str, str]
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for conversation in conversations:
        outputs: list[dict[str, Any]] = []
        for text_index, text in enumerate(conversation.texts):
            candidate_paths = [
                successful_paths.get(
                    candidate_id(conversation.index, text_index, reference_index)
                )
                for reference_index in range(len(conversation.reference_audio_paths))
            ]
            outputs.append(
                {
                    "text": text,
                    "candidate_audio_path": candidate_paths,
                    "reference_audio_path": conversation.reference_audio_paths,
                    "reference_audio_text": conversation.reference_audio_texts,
                }
            )
        payload.append({"id": conversation.input_id, "output": outputs})
    return payload


def failed_json(
    failed_rows: Sequence[dict[str, str]], candidates_by_id: dict[str, Candidate]
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in failed_rows:
        candidate = candidates_by_id[row["row_id"]]
        payload.append(
            {
                "candidate_id": candidate.candidate_id,
                "id": candidate.input_id,
                "text_index": candidate.text_index,
                "reference_index": candidate.reference_index,
                "text": candidate.text,
                "reference_audio_path": candidate.reference_audio_path,
                "reference_audio_text": candidate.reference_audio_text,
                "error": row["error"],
            }
        )
    return payload


def write_outputs(
    output_dir: Path,
    conversations: Sequence[Conversation],
    candidates_by_id: dict[str, Candidate],
    generated_rows: Sequence[dict[str, str]],
    failed_rows: Sequence[dict[str, str]],
) -> None:
    atomic_write_text(
        output_dir / "generated.tsv", tsv_content(GENERATED_FIELDS, generated_rows)
    )
    atomic_write_text(
        output_dir / "failed.tsv", tsv_content(FAILED_FIELDS, failed_rows)
    )
    successful_paths = {row["id"]: row["speechpath"] for row in generated_rows}
    atomic_write_json(
        output_dir / "generated.json",
        grouped_generated_json(conversations, successful_paths),
    )
    atomic_write_json(
        output_dir / "failed.json", failed_json(failed_rows, candidates_by_id)
    )


def manifest_payload(
    conversations: Sequence[Conversation], candidates: Sequence[Candidate]
) -> dict[str, Any]:
    return {
        "input": [asdict(conversation) for conversation in conversations],
        "expanded": [asdict(candidate) for candidate in candidates],
    }


def validate_resume_rows(
    output_dir: Path,
    candidates_by_id: dict[str, Candidate],
    generated_rows: Sequence[dict[str, str]],
    failed_rows: Sequence[dict[str, str]],
) -> set[str]:
    completed: set[str] = set()
    for row in generated_rows:
        row_id = row["id"]
        candidate = candidates_by_id.get(row_id)
        if candidate is None or row_id in completed:
            raise ValueError(f"Invalid or duplicate completed candidate id: {row_id}")
        if row["text"] != candidate.text:
            raise ValueError(f"Existing metadata text mismatch for candidate {row_id}")
        wav_path = output_dir / row["speechpath"]
        if not wav_path.is_file():
            raise ValueError(f"Existing generated WAV is missing: {wav_path}")
        completed.add(row_id)
    for row in failed_rows:
        row_id = row["row_id"]
        if row_id not in candidates_by_id or row_id in completed:
            raise ValueError(f"Invalid or duplicate failed candidate id: {row_id}")
        completed.add(row_id)
    return completed


def default_model_loader(model_path: str, device: str, attention: str) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.float32 if device.casefold().startswith("cpu") else torch.bfloat16
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attention,
    )
    if getattr(model.model, "tts_model_type", None) != "base":
        raise ValueError(f"Model is not a Qwen3-TTS Base checkpoint: {model_path}")
    return model


def seed_candidate(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def generate_dataset(
    args: argparse.Namespace,
    model_loader: Callable[[str, str, str], Any] = default_model_loader,
    audio_writer: Callable[[Any, int, Path, int], None] = default_audio_writer,
    candidate_seeder: Callable[[int], None] = seed_candidate,
) -> None:
    input_bytes = args.input_json.read_bytes()
    conversations = load_conversations(args.input_json)
    candidates = expand_candidates(conversations)
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    output_dir = args.output_dir.resolve()
    expected_state = state_payload(args, input_bytes)

    generated_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    if output_dir.exists() and not args.overwrite:
        state_path = output_dir / ".generation_state.json"
        if not state_path.is_file():
            raise ValueError(
                f"Existing output has no generation state: {output_dir}. "
                "Use --overwrite to restart."
            )
        try:
            existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read generation state {state_path}: {exc}") from exc
        if existing_state != expected_state:
            raise ValueError(
                f"Existing output does not match current input/settings: {output_dir}. "
                "Use --overwrite to restart."
            )
        generated_rows = read_tsv(output_dir / "generated.tsv", GENERATED_FIELDS)
        failed_rows = read_tsv(output_dir / "failed.tsv", FAILED_FIELDS)

    completed = validate_resume_rows(
        output_dir, candidates_by_id, generated_rows, failed_rows
    )
    print(f"Input records: {len(conversations)}")
    print(f"Expanded candidates: {len(candidates)}")
    print(f"Completed candidates: {len(completed)}")
    if len(completed) == len(candidates):
        write_outputs(
            output_dir,
            conversations,
            candidates_by_id,
            generated_rows,
            failed_rows,
        )
        print("Generated candidates: 0")
        print(f"Skipped candidates: {len(completed)}")
        print(f"Failed candidates: {len(failed_rows)}")
        return

    model = model_loader(args.model_path, args.device, args.attn_implementation)
    if args.overwrite and output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(f"Refusing to overwrite unsafe output path: {output_dir}")
        shutil.rmtree(output_dir)
        generated_rows = []
        failed_rows = []
        completed = set()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(exist_ok=True)
    atomic_write_json(output_dir / ".generation_state.json", expected_state)
    atomic_write_json(
        output_dir / "expanded_manifest.json",
        manifest_payload(conversations, candidates),
    )
    write_outputs(
        output_dir, conversations, candidates_by_id, generated_rows, failed_rows
    )

    prompt_cache: dict[tuple[str, str], Any] = {}
    gen_kwargs = generation_options(args)
    generated_this_run = 0
    failures_this_run = 0
    for candidate in candidates:
        if candidate.candidate_id in completed:
            continue
        try:
            candidate_seeder(args.seed + candidate.ordinal)
            cache_key = (
                candidate.reference_audio_path,
                candidate.reference_audio_text,
            )
            prompt = prompt_cache.get(cache_key)
            if prompt is None:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=candidate.reference_audio_path,
                    ref_text=candidate.reference_audio_text,
                    x_vector_only_mode=False,
                )
                prompt_cache[cache_key] = prompt
            wavs, source_rate = model.generate_voice_clone(
                text=candidate.text,
                language=args.language,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                **gen_kwargs,
            )
            if len(wavs) != 1:
                raise RuntimeError(
                    f"Expected one waveform, got {len(wavs)} for {candidate.candidate_id}"
                )
            relative_wav = f"wav/utt_{candidate.ordinal:06d}.wav"
            audio_writer(
                wavs[0],
                int(source_rate),
                output_dir / relative_wav,
                args.sample_rate,
            )
            generated_rows.append(
                {
                    "speechpath": relative_wav,
                    "text": candidate.text,
                    "id": candidate.candidate_id,
                }
            )
            generated_this_run += 1
            print(
                f"Generated {candidate.candidate_id} "
                f"({candidate.ordinal + 1}/{len(candidates)})"
            )
        except Exception as exc:
            failed_rows.append(
                {
                    "row_id": candidate.candidate_id,
                    "id": candidate.input_id,
                    "text": candidate.text,
                    "ref_audio": candidate.reference_audio_path,
                    "ref_audio_text": candidate.reference_audio_text,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures_this_run += 1
            print(f"Failed {candidate.candidate_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        write_outputs(
            output_dir,
            conversations,
            candidates_by_id,
            generated_rows,
            failed_rows,
        )
        if failures_this_run and args.on_error == "raise":
            raise RuntimeError(f"Generation failed for {candidate.candidate_id}")

    print(f"Generated candidates: {generated_this_run}")
    print(f"Skipped candidates: {len(completed)}")
    print(f"Failed candidates: {len(failed_rows)}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        generate_dataset(parse_args(argv))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
