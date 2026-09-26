"""Tests for the eval harness (eval/evaluate.py).

Run with:  uv run pytest -v
No API key is required: mock mode injects a fake LLM that echoes ground truth.
"""

from eval.evaluate import (
    annotation,
    evaluate,
    load_split,
    parse_args,
    strip_label,
    _text_match,
)


def test_strip_label_drops_prefix():
    assert strip_label("Invoice Date: 10-Sep-2016") == "10-Sep-2016"
    assert strip_label("GSTIN: OG@AAMFCO376K124") == "OG@AAMFCO376K124"
    assert strip_label("no label here") == "no label here"


def test_annotation_handles_object_and_empty_list():
    raw = {
        "DATE": {"bbox": [0, 0], "text": "Date: 2020-01-01"},
        "INVOICE_INFO": [],
        "LOGO": {"bbox": [0, 0]},
    }
    out = annotation(raw)
    assert out["DATE"] == "2020-01-01"
    assert out["INVOICE_INFO"] == ""
    assert "LOGO" not in out


def test_text_match_exact_and_substring():
    assert _text_match("10-Sep-2016", "10-Sep-2016")
    # label-prefixed model output still matches
    assert _text_match("Invoice Date: 10-Sep-2016", "10-Sep-2016")
    assert _text_match("og@aamfco376k124", "OG@AAMFCO376K124")
    assert not _text_match("10-Sep-2016", "11-Oct-2016")


def test_load_split_default_path(tmp_path):
    # default split is strat1_test.csv; just assert it returns rows + honours limit
    args = parse_args(["--limit", "3"])
    rows = load_split(args)
    assert len(rows) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in rows)


def test_mock_mode_reports_perfect_match(tmp_path):
    report_path = tmp_path / "report.json"
    args = parse_args(
        ["--mode", "mock", "--limit", "3", "--quiet", "--report", str(report_path)]
    )
    report = evaluate(args)
    assert report["evaluated"] == 3
    assert report["failures"] == 0
    assert report["instance_exact_match_rate"] == 1.0
    assert report["non_exact_instance_rate"] == 0.0
    assert report_path.exists()
