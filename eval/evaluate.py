"""Quality evaluation for the document parser.

For each instance in an eval split we:
  1. load the image and its ground-truth annotation,
  2. convert the image to PDF bytes,
  3. run the extraction pipeline (in-process against the configured LLM, or
     against a running server), and
  4. score the extracted fields against the annotated values.

Ground-truth annotations store values as "<Label>: <value>" (e.g.
`"INVOICE ID 5Y3M7d-590"` -> the text on the right of the last ":"). The parser
is asked to return just the value, so we strip the label off the annotation
before comparing.

Modes
-----
  service  run the real pipeline in-process (DocumentParserService + OpenAILLM).
           Requires OPENAI_API_KEY / OPENAI_BASE_URL in the environment.
  http     send the image to a running server's /extract/json endpoint.
  mock     inject a fake LLM that echoes the ground truth, so the dataset,
           image->PDF rendering and scoring can be exercised with no network
           cost (expect ~100% on scalar fields).

Examples
--------
  uv run python eval/evaluate.py --mode mock --limit 15
  uv run python eval/evaluate.py --split-csv eval/dataset/strat1_test.csv --limit 100
  uv run python eval/evaluate.py --mode http --base-url http://localhost:8000 --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

logger = logging.getLogger("eval")

DATASET_ROOT = Path("eval/dataset")
IMAGES_DIR = DATASET_ROOT / "images"
ORIGINALS_DIR = DATASET_ROOT / "original"
SPLIT1 = DATASET_ROOT / "strat1_test.csv"
SPLIT2 = DATASET_ROOT / "Strat2_Split.txt"


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_rows_from_csv(csv_path: Path) -> List[tuple[str, str]]:
    with csv_path.open(newline="") as fh:
        return [(r["img_path"], r["annot_path"]) for r in csv.DictReader(fh)]


def split_rows_from_strat2(path: Path) -> List[tuple[str, str]]:
    """Template-level split.

    `Strat2_Split.txt` lists template indices for train/dev/test. We treat the
    test indices as template numbers and iterate their 200 instances each, so
    the returned rows are the held-out templates' instances.
    """
    text = path.read_text()
    m = re.search(r"test_inds\s*=\s*(\[.*\])", text, re.S)
    if not m:
        raise ValueError(f"test_inds not found in {path}")
    inds: list[int] = json.loads(m.group(1))
    rows: List[tuple[str, str]] = []
    for t in inds:
        rows.extend((f"Template{t}_Instance{i}.jpg", f"Template{t}_Instance{i}.json") for i in range(200))
    return rows


def load_split(args: argparse.Namespace) -> List[tuple[str, str]]:
    if args.split2:
        rows = split_rows_from_strat2(SPLIT2)
    elif args.split_csv:
        rows = split_rows_from_csv(Path(args.split_csv))
    else:
        rows = split_rows_from_csv(SPLIT1)
    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(rows)
    return rows[: args.limit] if args.limit else rows


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #
def image_to_pdf_bytes(image_path: Path) -> bytes:
    """Render a JPEG image to a single-page PDF (the parser consumes PDF bytes)."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF", dpi=[300.0, 300.0])
    return buf.getvalue()


def strip_label(text: str) -> str:
    """Drop the field label prefix ("Date: 18-Jan-1999" -> "18-Jan-1999")."""
    text = text.strip()
    return text.split(":", 1)[1].strip() if ":" in text else text


def annotation(gt_raw: Any) -> Dict[str, str]:
    """Return {field: value} for the scalar string fields of an annotation.

    Non-empty scalar fields are objects of the form {"bbox": [...], "text": "<label>: <value>"}.
    Labels are stripped. Empty list values mean "no value expected"; structured
    values (non-empty lists like TABLE) and label-less bbox-only objects are
    skipped from text scoring.
    """
    out: Dict[str, str] = {}
    for key, val in gt_raw.items():
        if isinstance(val, dict) and "text" in val:
            out[key] = strip_label(val["text"])
        elif isinstance(val, list) and not val:
            out[key] = ""
    return out


def expected_annotation(annot_path: Path) -> Optional[Dict[str, str]]:
    try:
        return annotation(json.loads(annot_path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None


def schema_from_fields(field_names: Sequence[str]):
    """Schema the pipeline is asked to extract (fields echoed as the keys)."""
    from backend.models import FieldSchema, Schema

    return Schema(name="eval", fields=[FieldSchema(name=n, type="any") for n in field_names])


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Returns the ground-truth JSON, so the harness runs end-to-end offline."""

    def __init__(self, expected: Dict[str, str]) -> None:
        self._expected = expected

    def chat_completion(self, messages: Any) -> str:
        return json.dumps(self._expected)


def run_http(client: Any, url: str, pdf_bytes: bytes, schema: Any) -> Dict[str, Any]:
    """Send one image to a running server's /extract/json endpoint."""
    body = {
        "file_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "schema": schema.model_dump(),
    }
    resp = client.post(f"{url.rstrip('/')}/extract/json", json=body, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"server returned status {payload.get('status')!r}")
    return payload.get("data", {})


async def _run_parse(parser: Any, doc: Any) -> Dict[str, Any]:
    """Await a single extraction call (the service parse() is now a coroutine)."""
    return await parser.parse(doc)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _text_match(got: str, expected: str) -> bool:
    """True for an exact (normalised) match, or a label/subset substring match.

    A model may return the value with its label intact ("Invoice Date: 10-Sep")
    or only part of a longer value; treat either as correct.
    """
    if norm(got) == norm(expected):
        return True
    pn, en = norm(got), norm(expected)
    return bool(pn) and (en in pn or pn in en)


@dataclass
class RunStats:
    evaluated: int = 0
    exact_instances: int = 0
    per_field: Dict[str, Dict[str, int]] = field(default_factory=dict)
    missing_fields: Dict[str, int] = field(default_factory=dict)
    missing_instances: int = 0
    per_instance: List[Dict[str, Any]] = field(default_factory=list)


def score(instance_id: str, expected: Dict[str, str], predicted: Dict[str, Any], stats: RunStats) -> None:
    stats.evaluated += 1
    pred = predicted or {}

    instance_exact = True
    for field_name, exp_val in expected.items():
        fs = stats.per_field.setdefault(field_name, {"matches": 0, "count": 0, "missing": 0})
        fs["count"] += 1
        got = pred.get(field_name)

        if exp_val == "":
            # Empty list in the annotation means "no value expected"; returning
            # nothing is a correct prediction.
            if not got:
                fs["matches"] += 1
            else:
                instance_exact = False
            continue

        if got is None or got == "":
            fs["missing"] += 1
            stats.missing_fields[field_name] = stats.missing_fields.get(field_name, 0) + 1
            instance_exact = False
            continue

        if _text_match(str(got), exp_val):
            fs["matches"] += 1
        else:
            instance_exact = False

    if instance_exact:
        stats.exact_instances += 1
    else:
        stats.missing_instances += 1
    stats.per_instance.append(
        {
            "instance": instance_id,
            "matched": sum(
                1 for f, ev in expected.items()
                if (ev != "" and pred.get(f) and _text_match(str(pred[f]), ev))
                or (ev == "" and not pred.get(f))
            ),
            "total": len(expected),
            "exact": instance_exact,
        }
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_parser(args: argparse.Namespace):
    """Return a parser ready to run, or the httpx client for --mode http."""
    if args.mode == "http":
        import httpx

        return httpx.Client(), None

    from backend.config import get_settings
    from backend.services import (
        DocumentParserService,
        PdfToImageConversionService,
    )
    from backend.services.llm import OpenAILLM

    settings = get_settings()
    parser = DocumentParserService(
        llm=None,
        pdf_to_image_conversion_service=PdfToImageConversionService(settings),
        settings=settings,
    )
    if args.mode == "service":
        parser.llm = OpenAILLM(settings)
    return None, parser


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rows = load_split(args)
    label = "strat2" if args.split2 else (args.split_csv or "strat1_test")
    logger.info("eval: mode=%s n=%d split=%s", args.mode, len(rows), label)

    runner, parser = build_parser(args)
    stats = RunStats()
    failures = 0
    start = time.time()
    for img_rel, annot_rel in rows:
        try:
            pdf_bytes = image_to_pdf_bytes(IMAGES_DIR / img_rel)
            expected = expected_annotation(ORIGINALS_DIR / annot_rel)
            if expected is None:
                failures += 1
                continue
            schema = schema_from_fields(expected)

            if args.mode in ("mock", "service"):
                from backend.services import Document as _Document

                if args.mode == "mock":
                    parser.llm = FakeLLM(expected)  # type: ignore[assignment]
                predicted = asyncio.run(
                    _run_parse(parser, _Document(pdf_bytes=pdf_bytes, schema=schema))
                )
            else:  # http
                predicted = run_http(runner, args.base_url, pdf_bytes, schema)
        except Exception as exc:  # keep the run going; record the failure
            failures += 1
            logger.info("eval: %s -> %s", img_rel, exc)
            continue

        score(img_rel, expected, predicted, stats)

    elapsed = time.time() - start
    report = build_report(args, len(rows), stats, elapsed, failures)
    print_report(report)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        logger.info("eval: report written to %s", args.report)
    return report


def build_report(args, n_rows, stats, elapsed, failures) -> Dict[str, Any]:
    n = stats.evaluated or 1
    per_field = {
        f: {**s, "accuracy": round(s["matches"] / s["count"], 4) if s["count"] else None}
        for f, s in stats.per_field.items()
    }
    return {
        "mode": args.mode,
        "split": "strat2" if args.split2 else (args.split_csv or "strat1_test"),
        "requested": n_rows,
        "evaluated": stats.evaluated,
        "failures": failures,
        "seconds": round(elapsed, 2),
        "per_second": round(stats.evaluated / elapsed, 3) if elapsed else 0.0,
        "instance_exact_match_rate": round(stats.exact_instances / n, 4),
        "non_exact_instance_rate": round(stats.missing_instances / n, 4),
        "per_field_accuracy": per_field,
    }


def print_report(report: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("Document parser eval")
    print("=" * 60)
    print(f"mode        : {report['mode']}")
    print(f"split       : {report['split']}")
    print(f"evaluated   : {report['evaluated']} / {report['requested']} "
          f"({report['failures']} failures)")
    print(f"elapsed     : {report['seconds']}s ({report['per_second']}/s)")
    print(f"instance exact-match rate : {report['instance_exact_match_rate']*100:.1f}%")
    print(f"non-exact-instance rate   : {report['non_exact_instance_rate']*100:.1f}%")
    print("-" * 60)
    print(f"{'field':<20}{'acc':>7}{'n':>6}")
    for f, s in sorted(report["per_field_accuracy"].items()):
        acc = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "n/a"
        print(f"{f:<20}{acc:>7}{s['count']:>6}")
    print("=" * 60)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the document parser quality.")
    p.add_argument("--mode", choices=["service", "http", "mock"], default="mock",
                   help="Where extraction runs (default: mock, for a cost-free check).")
    p.add_argument("--split-csv", help="Path to a strat1_*_split CSV (img_path,annot_path).")
    p.add_argument("--split2", action="store_true", help="Use the Strat2 template-level split.")
    p.add_argument("--limit", type=int, default=20,
                   help="Cap the number of instances (default: 20).")
    p.add_argument("--seed", type=int, default=None, help="Seed for shuffling the split.")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="Server URL for --mode http.")
    p.add_argument("--report", default="eval/eval_report.json", help="JSON report path.")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logging.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
