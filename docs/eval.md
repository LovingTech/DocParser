# Evaluations

The evaluation harness (`eval/evaluate.py`) measures how well the parser extracts a
defined schema across a labeled dataset. For each instance it runs the full extraction
pipeline against a rendered image, compares the result to the ground-truth annotation,
and reports per-field accuracy plus an instance-level exact-match rate.

```text
JPEG image ──▶  parse()  ──▶  JSON ──▶  annotate() ──▶  match against ground truth ──▶  report
```

## Modes

Three extraction backends share the exact same scoring code; pick the one that fits the
stage you're in.

| Mode      | What runs extraction                                   | Needs a running server | Needs `OPENAI_API_KEY` |
|-----------|--------------------------------------------------------|--------------------------|------------------------|
| `mock`    | A fake LLM in-process that echoes the ground truth     | No                       | No                     |
| `service` | The real LLM adapter injected into `DocumentParserService` in-process | No    | Yes                    |
| `http`    | A live `backend.main:app` served on `--base-url`       | Yes                      | Yes                    |

`mock` is the default. It's cost-free and returns a perfect score by construction — use
it as a smoke test / regression gate for the harness itself, never to measure model
quality.

## Dataset layout

All data lives under `eval/dataset/`:

| Path                            | Contents                                             |
|---------------------------------|------------------------------------------------------|
| `images/<Template>.jpg`         | One rendered JPEG per instance (the parser input).   |
| `original/<Template>.json`      | Ground-truth annotation for that instance.           |
| `strat1_train.csv`              | CSV split — `img_path,annot_path` columns (7500).    |
| `strat1_dev.csv`                | CSV split (1250).                                    |
| `strat1_test.csv`               | CSV split (1250).                                    |
| `Strat2_Split.txt`              | Template indices (`train_inds`/`dev_inds`/`test_inds`) for the `--split2` template-level split. |

Splits are selected with `--split2` (Strat2) or `--split-csv <path>` (any `strat1_*`
CSV). The default is `strat1_test.csv`. Every CSV is relative to `eval/dataset/`.

### Annotation format

Each `original/<Template>.json` maps field names to a value. Most fields are objects:

```json
{ "DATE":  { "bbox": [[328.0, 776.4], [470.7, 788.4]], "text": "Invoice Date: 10-Sep-2016" } }
```

Three shapes exist:

- **Object with `text`** — the scalar value to compare. The `text` is *label-prefixed*
  (`"Invoice Date: 10-Sep-2016"`); the harness strips everything up to the first `:`, so
  the model need only return the bare value (`"10-Sep-2016"`).
- **Empty list** (`"INVOICE_INFO": []`) — "no value expected"; scored as an empty string.
- **Structured list** (`"TABLE": [[{...}]]`) — skipped from text scoring entirely.

Objects that carry a `bbox` but **no** `text` (e.g. a `LOGO`) are dropped from scoring.

## How scoring works

- **Per field** — for every instance whose ground truth is not `None`, a field scores
  `True` when the model value is a **case-insensitive substring** of the ground truth
  (`"Invoice Date: 10-Sep-2016"` matches `"10-Sep-2016"`). `accuracy = matched / count`
  across the split (count excludes missing ground truth; structured fields count all
  instances).
- **Instance exact match** — an instance matches only when **every** scored field
  matched. `instance_exact_match_rate = exact_instances / n`.
- **Non-exact-instance rate** — the share of instances where at least one field failed.
- **Failures** — instances whose schema or annotation was missing/corrupt, or whose
  extraction raised; they are excluded from `n` and noted in the report.

## Running

```bash
# Smoke test — cost-free, fake LLM echoes ground truth, no API key.
uv run eval/evaluate.py --mode mock --limit 20

# Hit a real (served) pipeline in-process; needs OPENAI_API_KEY.
uv run eval/evaluate.py --mode service

# Hit a live server; needs the server up on the base URL.
uv run eval/evaluate.py --mode http --base-url http://localhost:8000
```

### Options

| Flag             | Default                         | Description                                              |
|------------------|---------------------------------|----------------------------------------------------------|
| `--mode`         | `mock`                          | `mock`, `service`, or `http`.                            |
| `--split-csv`    | `eval/dataset/strat1_test.csv`  | Path to a `strat1_*` split CSV.                          |
| `--split2`       | `False`                         | Use the `Strat2_Split.txt` template-level split.         |
| `--limit`        | `20`                            | Cap the number of instances.                             |
| `--seed`         | *(none)*                        | Seed for shuffling the split.                            |
| `--base-url`     | `http://localhost:8000`         | Server URL for `--mode http`.                            |
| `--report`       | `eval/eval_report.json`         | Where to write the JSON report.                          |
| `--quiet`        | `False`                         | Suppress INFO logging.                                   |

## Output

A human-readable table prints to stdout, and the same figures are written to the report
file (`--report`, default `eval/eval_report.json`):

```json
{
  "mode": "service",
  "split": "strat1_test",
  "requested": 1250,
  "evaluated": 1248,
  "failures": 2,
  "seconds": 412.5,
  "per_second": 3.024,
  "instance_exact_match_rate": 0.862,
  "non_exact_instance_rate": 0.138,
  "per_field_accuracy": { "TOTAL": { "matches": 1138, "count": 1248, "missing": 110, "accuracy": 0.912 } }
}
```

`accuracy` is `None` for fields with zero scored instances.

## Testing

Harness behavior (label stripping, annotation parsing, substring matching, split loading)
is covered by `tests/test_eval.py`. Those tests require no API key and run in `mock` mode.
