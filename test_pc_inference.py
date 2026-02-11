"""
Unit and integration tests for pc_inference.py

Requires: pytest, numpy, tflite-runtime or tensorflow
Run:      pytest test_pc_inference.py -v
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
import pc_inference  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def ai():
    """Module-scoped FisheryAI instance (loads the real model once)."""
    return pc_inference.FisheryAI()


@pytest.fixture
def topics(ai):
    """List of valid topic names from the loaded model."""
    return list(ai.topic_map.keys())


@pytest.fixture
def sample_data():
    """A plausible 5-year data window."""
    return [15000, 18000, 17000, 16500, 19000]


@pytest.fixture
def tmp_meta(tmp_path):
    """Create a minimal valid model_meta.json for constructor tests."""
    # We need the real meta to build fixtures - read it once
    real_meta_path = os.path.join(
        os.path.dirname(__file__), "model_meta.json"
    )
    with open(real_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


# ===========================================================================
# 1. Pure utility functions
# ===========================================================================

class TestSparkline:
    def test_ascending(self):
        result = pc_inference.sparkline([1, 2, 3, 4, 5])
        assert len(result) == 5
        # First char should be lowest block, last should be highest
        assert result[0] < result[-1]

    def test_descending(self):
        result = pc_inference.sparkline([5, 4, 3, 2, 1])
        assert len(result) == 5
        assert result[0] > result[-1]

    def test_flat(self):
        result = pc_inference.sparkline([42, 42, 42])
        assert len(result) == 3
        # All same value → all same block (middle block)
        assert len(set(result)) == 1

    def test_empty(self):
        assert pc_inference.sparkline([]) == ""

    def test_single_value(self):
        result = pc_inference.sparkline([100])
        assert len(result) == 1

    def test_two_values(self):
        result = pc_inference.sparkline([0, 100])
        assert len(result) == 2
        assert result[0] != result[1]


class TestFormatTrendArrow:
    def test_upward(self):
        assert "[YUKARI]" in pc_inference.format_trend_arrow([100, 200])

    def test_downward(self):
        assert "[ASAGI]" in pc_inference.format_trend_arrow([200, 100])

    def test_flat(self):
        assert "[YATAY]" in pc_inference.format_trend_arrow([100, 105])

    def test_single_value(self):
        assert pc_inference.format_trend_arrow([100]) == ""

    def test_empty(self):
        assert pc_inference.format_trend_arrow([]) == ""

    def test_threshold_boundary_up(self):
        # Exactly 10% increase → should be YATAY (not strictly >10%)
        assert "[YATAY]" in pc_inference.format_trend_arrow([100, 110])

    def test_threshold_boundary_down(self):
        # Exactly -10% → should be YATAY
        assert "[YATAY]" in pc_inference.format_trend_arrow([100, 90])

    def test_just_over_threshold(self):
        # 11% increase → YUKARI
        assert "[YUKARI]" in pc_inference.format_trend_arrow([100, 111])


class TestTypingEffect:
    def test_no_animation(self, capsys):
        pc_inference.typing_effect("hello", animate=False)
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_animation_output(self, capsys):
        # With animate=True but speed=0 for fast execution
        pc_inference.typing_effect("hi", speed=0, animate=True)
        captured = capsys.readouterr()
        assert "hi" in captured.out


class TestSaveReport:
    def test_creates_file(self, tmp_path):
        filepath = str(tmp_path / "report.txt")
        pc_inference.save_report(filepath, "test_topic", [1, 2, 3], "rapor", 1.23)
        assert os.path.exists(filepath)

    def test_content(self, tmp_path):
        filepath = str(tmp_path / "report.txt")
        pc_inference.save_report(filepath, "avcilik", [10, 20, 30, 40, 50], "artis seyretti", 0.5)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "AVCILIK" in content
        assert "artis seyretti" in content
        assert "0.50ms" in content

    def test_appends(self, tmp_path):
        filepath = str(tmp_path / "report.txt")
        pc_inference.save_report(filepath, "t1", [1], "r1", 1.0)
        pc_inference.save_report(filepath, "t2", [2], "r2", 2.0)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2


class TestShowHistory:
    def test_empty_history(self, capsys):
        pc_inference.show_history([])
        captured = capsys.readouterr()
        assert "Henuz analiz yapilmadi" in captured.out

    def test_with_entries(self, capsys):
        history = [
            {
                "topic": "tüketim",
                "data": [10, 20, 30, 40, 50],
                "report": "artis seyretti",
                "ms": 1.5,
                "time": "12:00:00",
            }
        ]
        pc_inference.show_history(history)
        captured = capsys.readouterr()
        assert "OTURUM GECMISI (1 analiz)" in captured.out
        assert "TÜKETIM" in captured.out


class TestClearScreen:
    def test_outputs_ansi(self, capsys):
        pc_inference.clear_screen()
        captured = capsys.readouterr()
        assert "\033[2J" in captured.out
        assert "\033[H" in captured.out


# ===========================================================================
# 2. FisheryAI constructor validation
# ===========================================================================

class TestFisheryAIInit:
    def test_default_loads_successfully(self, ai):
        assert ai.interpreter is not None
        assert ai.word_index is not None
        assert ai.topic_map is not None

    def test_missing_model_file(self, tmp_path):
        fake_model = str(tmp_path / "nonexistent.tflite")
        real_meta = os.path.join(os.path.dirname(__file__), "model_meta.json")
        with pytest.raises(FileNotFoundError, match="Gerekli dosyalar bulunamadi"):
            pc_inference.FisheryAI(model_path=fake_model, meta_path=real_meta)

    def test_missing_meta_file(self, tmp_path):
        real_model = os.path.join(os.path.dirname(__file__), "fishery_model.tflite")
        fake_meta = str(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError, match="Gerekli dosyalar bulunamadi"):
            pc_inference.FisheryAI(model_path=real_model, meta_path=fake_meta)

    def test_both_files_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Gerekli dosyalar bulunamadi"):
            pc_inference.FisheryAI(
                model_path=str(tmp_path / "a.tflite"),
                meta_path=str(tmp_path / "b.json"),
            )

    def test_invalid_json_meta(self, tmp_path):
        real_model = os.path.join(os.path.dirname(__file__), "fishery_model.tflite")
        bad_meta = tmp_path / "bad.json"
        bad_meta.write_text("NOT VALID JSON {{{{", encoding="utf-8")
        with pytest.raises(ValueError, match="gecersiz JSON"):
            pc_inference.FisheryAI(model_path=real_model, meta_path=str(bad_meta))

    def test_missing_tokenizer_key(self, tmp_path, tmp_meta):
        real_model = os.path.join(os.path.dirname(__file__), "fishery_model.tflite")
        meta_no_tok = {k: v for k, v in tmp_meta.items() if k != "tokenizer_json"}
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps(meta_no_tok), encoding="utf-8")
        with pytest.raises(ValueError, match="beklenen anahtar"):
            pc_inference.FisheryAI(model_path=real_model, meta_path=str(meta_path))

    def test_missing_topic_map(self, tmp_path, tmp_meta):
        real_model = os.path.join(os.path.dirname(__file__), "fishery_model.tflite")
        meta_no_topic = {k: v for k, v in tmp_meta.items() if k != "topic_map"}
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps(meta_no_topic), encoding="utf-8")
        with pytest.raises(ValueError, match="topic_map"):
            pc_inference.FisheryAI(model_path=real_model, meta_path=str(meta_path))


# ===========================================================================
# 3. Input validation in analyze()
# ===========================================================================

class TestAnalyzeValidation:
    def test_wrong_length_short(self, ai, topics):
        with pytest.raises(ValueError, match="5 veri noktasi"):
            ai.analyze([1, 2, 3], topics[0])

    def test_wrong_length_long(self, ai, topics):
        with pytest.raises(ValueError, match="5 veri noktasi"):
            ai.analyze([1, 2, 3, 4, 5, 6], topics[0])

    def test_empty_list(self, ai, topics):
        with pytest.raises(ValueError, match="5 veri noktasi"):
            ai.analyze([], topics[0])

    def test_not_a_list(self, ai, topics):
        with pytest.raises(ValueError):
            ai.analyze("12345", topics[0])

    def test_nan_value(self, ai, topics):
        with pytest.raises(ValueError, match="NaN/Inf"):
            ai.analyze([1, 2, float("nan"), 4, 5], topics[0])

    def test_inf_value(self, ai, topics):
        with pytest.raises(ValueError, match="NaN/Inf"):
            ai.analyze([1, 2, float("inf"), 4, 5], topics[0])

    def test_negative_inf(self, ai, topics):
        with pytest.raises(ValueError, match="NaN/Inf"):
            ai.analyze([1, 2, float("-inf"), 4, 5], topics[0])

    def test_non_numeric_element(self, ai, topics):
        with pytest.raises(TypeError, match="sayisal degil"):
            ai.analyze([1, 2, "three", 4, 5], topics[0])

    def test_none_element(self, ai, topics):
        with pytest.raises(TypeError, match="sayisal degil"):
            ai.analyze([1, 2, None, 4, 5], topics[0])

    def test_invalid_topic(self, ai):
        with pytest.raises(ValueError, match="Gecersiz konu"):
            ai.analyze([1, 2, 3, 4, 5], "nonexistent_topic")

    def test_empty_topic(self, ai):
        with pytest.raises(ValueError, match="Gecersiz konu"):
            ai.analyze([1, 2, 3, 4, 5], "")


# ===========================================================================
# 4. Integration tests — actual model inference
# ===========================================================================

class TestAnalyzeIntegration:
    """These tests run real model inference and verify output shape/type."""

    def test_returns_tuple(self, ai, topics, sample_data):
        result = ai.analyze(sample_data, topics[0])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_report_is_string(self, ai, topics, sample_data):
        report, _ = ai.analyze(sample_data, topics[0])
        assert isinstance(report, str)
        assert len(report) > 0

    def test_inference_time_positive(self, ai, topics, sample_data):
        _, ms = ai.analyze(sample_data, topics[0])
        assert isinstance(ms, float)
        assert ms > 0

    def test_all_topics_produce_output(self, ai, topics):
        data = [100, 90, 80, 70, 60]
        for topic in topics:
            report, ms = ai.analyze(data, topic)
            assert isinstance(report, str), f"Topic '{topic}' did not return a string"
            assert len(report) > 0, f"Topic '{topic}' returned empty report"
            assert ms > 0, f"Topic '{topic}' had non-positive inference time"

    def test_constant_input(self, ai, topics):
        """When all 5 values are the same, model should still produce output."""
        report, ms = ai.analyze([50, 50, 50, 50, 50], topics[0])
        assert isinstance(report, str)
        assert len(report) > 0

    def test_ascending_data(self, ai, topics):
        report, _ = ai.analyze([10, 20, 30, 40, 50], topics[0])
        assert isinstance(report, str)

    def test_descending_data(self, ai, topics):
        report, _ = ai.analyze([50, 40, 30, 20, 10], topics[0])
        assert isinstance(report, str)

    def test_tuple_input(self, ai, topics):
        """analyze() should accept tuples as well as lists."""
        report, ms = ai.analyze((10, 20, 30, 40, 50), topics[0])
        assert isinstance(report, str)

    def test_float_data(self, ai, topics):
        """Floating-point data should work."""
        report, _ = ai.analyze([15.5, 16.2, 14.8, 17.1, 15.9], topics[0])
        assert isinstance(report, str)

    def test_negative_values(self, ai, topics):
        """Negative values are unusual for fisheries data but should not crash."""
        report, _ = ai.analyze([-5, -3, -1, 0, 2], topics[0])
        assert isinstance(report, str)

    def test_very_large_values(self, ai, topics):
        """Extremely large values should not crash."""
        report, _ = ai.analyze([1e9, 2e9, 3e9, 4e9, 5e9], topics[0])
        assert isinstance(report, str)

    def test_very_small_values(self, ai, topics):
        """Very small (near-zero) values should not crash."""
        report, _ = ai.analyze([0.001, 0.002, 0.003, 0.004, 0.005], topics[0])
        assert isinstance(report, str)

    def test_zero_values(self, ai, topics):
        """All-zero input should not crash."""
        report, _ = ai.analyze([0, 0, 0, 0, 0], topics[0])
        assert isinstance(report, str)

    def test_deterministic_output(self, ai, topics):
        """Same input should produce same output (model is deterministic)."""
        data = [100, 200, 300, 400, 500]
        topic = topics[0]
        report1, _ = ai.analyze(data, topic)
        report2, _ = ai.analyze(data, topic)
        assert report1 == report2


# ===========================================================================
# 5. Batch CLI via subprocess
# ===========================================================================

class TestBatchCLI:
    """Test the --batch mode via subprocess."""

    _script = os.path.join(os.path.dirname(__file__), "pc_inference.py")

    def test_batch_topic_1(self):
        result = subprocess.run(
            [sys.executable, self._script, "--batch", "1", "100,200,300,400,500"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "ms)" in result.stdout

    def test_batch_topic_4(self):
        result = subprocess.run(
            [sys.executable, self._script, "--batch", "4", "70,65,55,48,40"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_batch_invalid_topic(self):
        result = subprocess.run(
            [sys.executable, self._script, "--batch", "99", "1,2,3,4,5"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "HATA" in result.stderr

    def test_batch_wrong_data_count(self):
        result = subprocess.run(
            [sys.executable, self._script, "--batch", "1", "1,2,3"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "HATA" in result.stderr

    def test_batch_non_numeric_data(self):
        result = subprocess.run(
            [sys.executable, self._script, "--batch", "1", "a,b,c,d,e"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_batch_output_file(self, tmp_path):
        outfile = str(tmp_path / "out.txt")
        result = subprocess.run(
            [sys.executable, self._script,
             "--batch", "1", "100,200,300,400,500",
             "--output", outfile],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert os.path.exists(outfile)
        with open(outfile, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0

    def test_batch_no_animation(self):
        result = subprocess.run(
            [sys.executable, self._script,
             "--batch", "2", "50,60,70,80,90",
             "--no-animation"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, self._script, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "FisheryAI" in result.stdout


# ===========================================================================
# 6. parse_args
# ===========================================================================

class TestParseArgs:
    def test_default_args(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["pc_inference.py"])
        args = pc_inference.parse_args()
        assert args.batch is None
        assert args.output is None
        assert args.no_animation is False

    def test_batch_args(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["pc_inference.py", "--batch", "1", "1,2,3,4,5"])
        args = pc_inference.parse_args()
        assert args.batch == ["1", "1,2,3,4,5"]

    def test_output_args(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["pc_inference.py", "-o", "report.txt"])
        args = pc_inference.parse_args()
        assert args.output == "report.txt"

    def test_no_animation_flag(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["pc_inference.py", "--no-animation"])
        args = pc_inference.parse_args()
        assert args.no_animation is True


# ===========================================================================
# 7. Module-level constants & structure
# ===========================================================================

class TestModuleConstants:
    def test_width(self):
        assert pc_inference.WIDTH == 60

    def test_topic_info_keys(self):
        for key in pc_inference.TOPIC_INFO:
            info = pc_inference.TOPIC_INFO[key]
            assert "tanim" in info
            assert "birim" in info
            assert "ornek" in info

    def test_topic_info_matches_topic_map(self, ai):
        """All topics in topic_map should have entries in TOPIC_INFO."""
        for topic in ai.topic_map:
            assert topic in pc_inference.TOPIC_INFO, (
                f"Topic '{topic}' in topic_map but missing from TOPIC_INFO"
            )
