import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("generate_tts.py")
SPEC = importlib.util.spec_from_file_location("generate_tts_demo", MODULE_PATH)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demo)


class FakeModel:
    def __init__(self, supported=None):
        self.supported = supported or ["vivian", "ryan"]
        self.calls = []

    def get_supported_speakers(self):
        return self.supported

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [[0.0, 0.1]], 24000


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_path = self.root / "input.json"
        self.speaker_path = self.root / "speakers.txt"
        self.output_path = self.root / "output"

    def tearDown(self):
        self.temp.cleanup()

    def args(self, overwrite=False, sample_rate=16000):
        return Namespace(
            input_json=self.input_path,
            speaker_file=self.speaker_path,
            output_dir=self.output_path,
            sample_rate=sample_rate,
            model_path="test-model",
            device="cuda:0",
            language="Auto",
            overwrite=overwrite,
        )

    def write_inputs(self, value=None):
        if value is None:
            value = [
                {"text": "Hello", "prompt": "Joyful"},
                {"text": "Goodbye", "prompt": "Calm"},
            ]
        self.input_path.write_text(json.dumps(value), encoding="utf-8")

    def writer(self, calls):
        def write(wav, source_rate, target, target_rate):
            calls.append((wav, source_rate, target, target_rate))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-wav")
        return write

    def test_single_object_is_accepted(self):
        self.write_inputs({"text": "Hello", "prompt": "Joyful"})
        self.assertEqual(demo.load_inputs(self.input_path)[0]["text"], "Hello")

    def test_invalid_input_and_duplicate_speaker_are_rejected(self):
        self.write_inputs({"text": "Hello"})
        with self.assertRaisesRegex(ValueError, "prompt"):
            demo.load_inputs(self.input_path)
        self.speaker_path.write_text("Vivian\nvivian\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            demo.load_speakers(self.speaker_path)

    def test_generates_cartesian_dataset_and_metadata(self):
        self.write_inputs()
        self.speaker_path.write_text("Vivian\nRyan\n", encoding="utf-8")
        model = FakeModel()
        writes = []
        demo.generate_dataset(self.args(), lambda *_: model, self.writer(writes))

        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(writes), 4)
        self.assertTrue((self.output_path / "Vivian/wav/000001.wav").is_file())
        rows = demo.read_metadata(self.output_path / "Vivian/metadata.jsonl")
        self.assertEqual(
            rows,
            [
                {"audiofile_path": "wav/000001.wav", "text": "Hello"},
                {"audiofile_path": "wav/000002.wav", "text": "Goodbye"},
            ],
        )
        self.assertEqual(model.calls[0]["instruct"], "Joyful")
        self.assertTrue(all(write[3] == 16000 for write in writes))

    def test_completed_run_skips_without_loading_model(self):
        self.write_inputs()
        self.speaker_path.write_text("Vivian\n", encoding="utf-8")
        first_model = FakeModel()
        demo.generate_dataset(self.args(), lambda *_: first_model, self.writer([]))

        def unexpected_loader(*_):
            raise AssertionError("completed resume should not load the model")

        demo.generate_dataset(self.args(), unexpected_loader, self.writer([]))
        self.assertEqual(len(first_model.calls), 2)

    def test_partial_run_resumes_from_first_missing_wav(self):
        self.write_inputs()
        self.speaker_path.write_text("Vivian\n", encoding="utf-8")
        demo.generate_dataset(self.args(), lambda *_: FakeModel(), self.writer([]))
        (self.output_path / "Vivian/wav/000002.wav").unlink()

        model = FakeModel()
        demo.generate_dataset(self.args(), lambda *_: model, self.writer([]))
        self.assertEqual([call["text"] for call in model.calls], ["Goodbye"])

    def test_changed_settings_require_overwrite(self):
        self.write_inputs()
        self.speaker_path.write_text("Vivian\n", encoding="utf-8")
        demo.generate_dataset(self.args(), lambda *_: FakeModel(), self.writer([]))
        with self.assertRaisesRegex(ValueError, "does not match"):
            demo.generate_dataset(
                self.args(sample_rate=22050), lambda *_: FakeModel(), self.writer([])
            )

    def test_overwrite_regenerates_all_items(self):
        self.write_inputs()
        self.speaker_path.write_text("Vivian\n", encoding="utf-8")
        demo.generate_dataset(self.args(), lambda *_: FakeModel(), self.writer([]))
        model = FakeModel()
        demo.generate_dataset(self.args(overwrite=True), lambda *_: model, self.writer([]))
        self.assertEqual(len(model.calls), 2)

    def test_unsupported_speaker_fails_before_output(self):
        self.write_inputs()
        self.speaker_path.write_text("John\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported speakers"):
            demo.generate_dataset(self.args(), lambda *_: FakeModel(), self.writer([]))
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
