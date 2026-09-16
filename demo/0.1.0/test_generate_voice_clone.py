import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("generate_voice_clone.py")
SPEC = importlib.util.spec_from_file_location("qwen_clone_demo", MODULE_PATH)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class FakeModel:
    def __init__(self, fail_text=None):
        self.fail_text = fail_text
        self.prompt_calls = []
        self.generate_calls = []

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return [kwargs["ref_audio"]]

    def generate_voice_clone(self, **kwargs):
        self.generate_calls.append(kwargs)
        if kwargs["text"] == self.fail_text:
            raise RuntimeError("synthetic failure")
        return [[0.0, 0.1]], 24000


class VoiceCloneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ref_a = self.root / "ref_a.wav"
        self.ref_b = self.root / "ref_b.wav"
        self.ref_a.write_bytes(b"a")
        self.ref_b.write_bytes(b"b")
        self.input_path = self.root / "input.json"
        self.output_path = self.root / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def write_input(self, texts=None, refs=None):
        texts = texts or ["Hello", "Goodbye"]
        refs = refs or [
            (self.ref_a.name, "Reference A"),
            (self.ref_b.name, "Reference B"),
        ]
        payload = [
            {
                "id": "sample",
                "text": texts,
                "reference_audio_path": [item[0] for item in refs],
                "reference_audio_text": [item[1] for item in refs],
            }
        ]
        self.input_path.write_text(json.dumps(payload), encoding="utf-8")

    def args(self, **overrides):
        values = dict(
            model_path=str(self.root / "base-model"),
            input_json=self.input_path,
            output_dir=self.output_path,
            device="cuda:0",
            language="English",
            sample_rate=16000,
            seed=7,
            attn_implementation="sdpa",
            on_error="skip",
            overwrite=False,
            do_sample=None,
            top_k=None,
            top_p=None,
            temperature=None,
            repetition_penalty=None,
            subtalker_do_sample=None,
            subtalker_top_k=None,
            subtalker_top_p=None,
            subtalker_temperature=None,
            max_new_tokens=None,
        )
        values.update(overrides)
        return Namespace(**values)

    @staticmethod
    def writer(calls):
        def write(wav, source_rate, target, target_rate):
            calls.append((source_rate, target_rate, target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"wav")
        return write

    def test_cross_product_order_and_relative_reference_paths(self):
        self.write_input()
        conversations = demo.load_conversations(self.input_path)
        candidates = demo.expand_candidates(conversations)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            [candidate.candidate_id for candidate in candidates],
            [
                "rec000000_text000000_ref000000",
                "rec000000_text000000_ref000001",
                "rec000000_text000001_ref000000",
                "rec000000_text000001_ref000001",
            ],
        )
        self.assertEqual(candidates[0].reference_audio_path, str(self.ref_a.resolve()))

    def test_schema_validation(self):
        self.input_path.write_text(
            json.dumps(
                [
                    {
                        "id": "bad",
                        "text": ["Hello"],
                        "reference_audio_path": [self.ref_a.name],
                        "reference_audio_text": ["Reference A", "Reference B"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            demo.load_conversations(self.input_path)

    def test_generation_groups_outputs_and_reuses_prompts(self):
        self.write_input()
        model = FakeModel()
        writes = []
        seeds = []
        demo.generate_dataset(
            self.args(),
            lambda *_: model,
            self.writer(writes),
            seeds.append,
        )
        self.assertEqual(len(model.generate_calls), 4)
        self.assertEqual(len(model.prompt_calls), 2)
        self.assertEqual(seeds, [7, 8, 9, 10])
        self.assertTrue(all(item[1] == 16000 for item in writes))
        grouped = json.loads((self.output_path / "generated.json").read_text())
        self.assertEqual(
            grouped[0]["output"][0]["candidate_audio_path"],
            ["wav/utt_000000.wav", "wav/utt_000001.wav"],
        )
        self.assertEqual(
            grouped[0]["output"][1]["candidate_audio_path"],
            ["wav/utt_000002.wav", "wav/utt_000003.wav"],
        )

    def test_completed_run_resumes_without_loading_model(self):
        self.write_input(texts=["Hello"], refs=[(self.ref_a.name, "Reference")])
        demo.generate_dataset(
            self.args(), lambda *_: FakeModel(), self.writer([]), lambda _: None
        )

        def unexpected_loader(*_):
            raise AssertionError("completed run should not load the model")

        demo.generate_dataset(
            self.args(), unexpected_loader, self.writer([]), lambda _: None
        )

    def test_changed_settings_require_overwrite(self):
        self.write_input(texts=["Hello"], refs=[(self.ref_a.name, "Reference")])
        demo.generate_dataset(
            self.args(), lambda *_: FakeModel(), self.writer([]), lambda _: None
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            demo.generate_dataset(
                self.args(sample_rate=22050),
                lambda *_: FakeModel(),
                self.writer([]),
                lambda _: None,
            )

    def test_skip_failure_preserves_alignment(self):
        self.write_input(texts=["Good", "Bad"], refs=[(self.ref_a.name, "Reference")])
        model = FakeModel(fail_text="Bad")
        demo.generate_dataset(
            self.args(), lambda *_: model, self.writer([]), lambda _: None
        )
        grouped = json.loads((self.output_path / "generated.json").read_text())
        self.assertEqual(grouped[0]["output"][0]["candidate_audio_path"], ["wav/utt_000000.wav"])
        self.assertEqual(grouped[0]["output"][1]["candidate_audio_path"], [None])
        failures = json.loads((self.output_path / "failed.json").read_text())
        self.assertEqual(failures[0]["text"], "Bad")

    def test_overwrite_regenerates_every_candidate(self):
        self.write_input(texts=["Hello"], refs=[(self.ref_a.name, "Reference")])
        demo.generate_dataset(
            self.args(), lambda *_: FakeModel(), self.writer([]), lambda _: None
        )
        model = FakeModel()
        demo.generate_dataset(
            self.args(overwrite=True),
            lambda *_: model,
            self.writer([]),
            lambda _: None,
        )
        self.assertEqual(len(model.generate_calls), 1)


if __name__ == "__main__":
    unittest.main()
