"""
Tests for the bug detector's windowing, prediction format and metrics.

    python -m pytest tests

These use a tiny randomly initialised encoder and need the CodeT5+ tokenizer
(downloaded from Hugging Face the first time); they are skipped without it.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from detect.evaluate import scores
from detect.model import LABELS, BugDetector, line_span, predict_many, window_with_line, windows

CODE = "module m(input a, output y);\n" + "".join(f"  wire w{i} = a;\n" for i in range(300)) + \
       "  assign y = !a;\nendmodule\n"


@pytest.fixture(scope="module")
def tok():
    try:
        from detect.model import load_tokenizer
        return load_tokenizer()
    except OSError as e:        # no network / no cache (other errors must fail)
        pytest.skip(f"CodeT5+ tokenizer not available: {e}")


@pytest.fixture(scope="module")
def tiny_model():
    cfg = transformers.T5Config(vocab_size=32200, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=2)
    torch.manual_seed(0)
    return BugDetector(transformers.T5EncoderModel(cfg)).eval()


def test_windows_cover_long_module(tok):
    wins = windows(tok, CODE, window=128, stride=96)
    assert len(wins) > 1
    assert all(len(ids) <= 128 for ids, _, _ in wins)
    assert wins[0][1] == 0
    assert wins[-1][2] >= len(CODE.rstrip()) - 1          # the tail is covered


def test_short_module_is_one_window(tok):
    assert len(windows(tok, "module m; endmodule\n")) == 1


def test_window_holds_bug_line(tok):
    line = CODE.splitlines().index("  assign y = !a;") + 1
    ids = window_with_line(tok, CODE, line, window=128, stride=96)
    assert "assign y" in tok.decode(ids)


def test_line_span():
    assert line_span("a\nb\nc\n", 0, 1) == (1, 1)
    assert line_span("a\nb\nc\n", 2, 5) == (2, 3)


def test_predict_format(tok, tiny_model):
    out = predict_many(tiny_model, tok, [CODE, "module m; endmodule\n"], batch_size=4)
    assert len(out) == 2
    for r in out:
        assert r["bug_type"] in LABELS
        assert r["label"] == ("clean" if r["bug_type"] == "none" else "buggy")
        assert abs(sum(r["probs"].values()) - 1) < 1e-3
        assert r["window_lines"][0] >= 1


def test_scores():
    s = scores(["none", "reset_polarity", "none"], ["none", "wrong_operator", "reset_polarity"])
    assert s["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["buggy_f1"] == pytest.approx(2 / 3, abs=1e-3)
