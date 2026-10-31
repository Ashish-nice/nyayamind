"""Evaluate generated legal text against the test-set ground truth.

The evaluator is deliberately independent of a particular model output file.
It aligns records by ``document_id`` (falling back to ``Case Name``), reads
large JSON arrays incrementally, and writes per-example plus cumulative
metrics.  By default it compares ``Explanation`` with
``Simplified_Reasoning``.  Use ``--prediction-field`` and
``--reference-field`` to evaluate another pair of fields.

Install the optional metric dependencies before running:

    pip install nltk rouge-score sacrebleu bert-score blanc
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Example:

    python code/evaluation/evaluate_text_metrics.py \
        --generated logs/gpt6_1_sol_350.json \
        --ground-truth data/test_data/test_v2.json \
        --output-dir evaluation_results/gpt6_1_sol

The output directory contains:

* ``per_example.csv``: all metric values for each aligned case.
* ``cumulative.csv``: running means after each case, useful for convergence
  plots.
* ``summary.json``: final means, counts, and configuration.

BLANC is a summary-helpfulness metric rather than a direct similarity metric.
It is reported with the ground truth as the source document and the generated
text as the summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


METRIC_NAMES = (
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu",
    "meteor",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "blanc",
)


def iter_json_array(path: Path, array_key: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array or an array in an envelope.

    This uses only the standard library, so evaluation remains usable even
    when ``ijson`` is not installed.  At most one parser buffer and one record
    are held at a time.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def read_more() -> None:
            nonlocal buffer, eof
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                eof = True

        read_more()
        while True:
            start = next((i for i, char in enumerate(buffer) if not char.isspace()), None)
            if start is not None:
                break
            if eof:
                raise ValueError(f"{path} is empty")
            buffer = ""
            read_more()

        if buffer[start] == "[":
            position = start + 1
        else:
            if not array_key:
                raise ValueError(f"{path} is an object; an array key is required")
            marker = json.dumps(array_key)
            while True:
                marker_position = buffer.find(marker)
                if marker_position >= 0:
                    position = buffer.find("[", marker_position + len(marker))
                    if position >= 0:
                        position += 1
                        break
                if eof:
                    raise ValueError(f"Could not find array key {array_key!r} in {path}")
                # Retain enough text to match a marker split across chunks.
                buffer = buffer[-len(marker) :]
                read_more()

        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"Unexpected end of array in {path}")
                buffer = ""
                position = 0
                read_more()

            if buffer[position] == "]":
                return

            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if not eof:
                        buffer = buffer[position:]
                        position = 0
                        read_more()
                        continue
                    raise ValueError(f"Invalid JSON in {path}: {error}") from error

            if not isinstance(record, dict):
                raise ValueError(f"Expected object records in {path}, got {type(record).__name__}")
            yield record
            position = end
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) and buffer[position] == ",":
                position += 1
            elif position < len(buffer) and buffer[position] != "]":
                raise ValueError(f"Expected comma or end of array in {path}")
            if position > 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def record_key(record: dict[str, Any], key_field: str) -> str:
    value = record.get(key_field)
    if value is None or str(value).strip() == "":
        value = record.get("Case Name", "")
    return str(value).strip()


def text_value(record: dict[str, Any], field: str) -> str:
    """Convert a configured field to text, including structured fields."""
    if field == "combined_reasoning":
        fields = (
            "Legal Issue Analysis",
            "Arguments of Petitioner",
            "Arguments of Respondent",
            "Deliberation",
            "Explanation",
        )
        return "\n\n".join(str(record.get(name, "") or "") for name in fields).strip()
    value = record.get(field, "")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "").strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def require_metrics(skip_blanc: bool) -> dict[str, Any]:
    """Import metric libraries with one actionable dependency error."""
    missing: list[str] = []
    modules: dict[str, Any] = {}
    imports = {
        "rouge_score": "rouge_score",
        "sacrebleu": "sacrebleu",
        "nltk": "nltk",
        "bert_score": "bert_score",
    }
    if not skip_blanc:
        imports["blanc"] = "blanc"
    for label, module_name in imports.items():
        try:
            modules[label] = importlib.import_module(module_name)
        except ImportError:
            missing.append(label)
    if missing:
        raise RuntimeError(
            "Missing metric dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install nltk rouge-score "
            "sacrebleu bert-score"
            + ("" if skip_blanc else " blanc")
        )
    modules["rouge_scorer"] = importlib.import_module("rouge_score.rouge_scorer")
    return modules


def lexical_metrics(candidate: str, reference: str, modules: dict[str, Any]) -> dict[str, float]:
    """Compute ROUGE, sentence BLEU, and METEOR for one pair."""
    scorer = modules["rouge_scorer"].RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge = scorer.score(reference, candidate)
    candidate_tokens = tokens(candidate)
    reference_tokens = tokens(reference)
    if not candidate_tokens or not reference_tokens:
        bleu = 0.0
        meteor = 0.0
    else:
        bleu = float(
            modules["sacrebleu"].sentence_bleu(
                candidate, [reference], smooth_method="exp"
            ).score
            / 100.0
        )
        try:
            meteor = float(
                modules["meteor_score"].meteor_score(
                    [reference_tokens], candidate_tokens
                )
            )
        except LookupError as error:
            raise RuntimeError(
                "NLTK WordNet data is missing. Run: "
                "python -c \"import nltk; nltk.download('wordnet'); "
                "nltk.download('omw-1.4')\""
            ) from error
    return {
        "rouge_1": float(rouge["rouge1"].fmeasure),
        "rouge_2": float(rouge["rouge2"].fmeasure),
        "rouge_l": float(rouge["rougeL"].fmeasure),
        "bleu": bleu,
        "meteor": meteor,
    }


def bertscore_metrics(
    candidates: Sequence[str], references: Sequence[str], modules: dict[str, Any]
) -> list[dict[str, float]]:
    """Compute BERTScore in one batch, preserving per-example values."""
    precision, recall, f1 = modules["bert_score"].score(
        list(candidates), list(references), lang="en", verbose=False
    )
    return [
        {
            "bertscore_precision": float(p),
            "bertscore_recall": float(r),
            "bertscore_f1": float(f),
        }
        for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist())
    ]


def blanc_score(candidate: str, reference: str, modules: dict[str, Any]) -> float:
    """Compute BLANC with the reference as document and candidate as summary."""
    try:
        evaluator = modules["blanc"].BlancHelp()
        value = evaluator.eval_once(candidate, reference)
    except AttributeError as error:
        raise RuntimeError(
            "The installed blanc package does not expose BlancHelp(). "
            "Install the PyPI package named 'blanc'."
        ) from error
    return float(value)


def build_reference_map(
    path: Path, key_field: str, wanted_keys: set[str], reference_field: str
) -> dict[str, str]:
    return {
        key: text_value(record, reference_field)
        for record in iter_json_array(path)
        if (key := record_key(record, key_field)) in wanted_keys
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-field", default="Explanation")
    parser.add_argument("--reference-field", default="Simplified_Reasoning")
    parser.add_argument("--key-field", default="document_id")
    parser.add_argument(
        "--no-blanc", action="store_true", help="Skip BLANC if only other metrics are needed"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="BERTScore batch size; lower this if GPU memory is limited",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modules = require_metrics(args.no_blanc)
    modules["meteor_score"] = importlib.import_module("nltk.translate.meteor_score")
    generated = [
        (record_key(record, args.key_field), text_value(record, args.prediction_field))
        for record in iter_json_array(args.generated, array_key="results")
    ]
    wanted = {key for key, _ in generated}
    references = build_reference_map(
        args.ground_truth, args.key_field, wanted, args.reference_field
    )
    pairs = [
        (key, candidate, references[key])
        for key, candidate in generated
        if key in references
    ]
    if not pairs:
        raise RuntimeError("No generated records matched the ground truth by the selected key")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = args.output_dir / "per_example.csv"
    cumulative_path = args.output_dir / "cumulative.csv"
    totals = {name: 0.0 for name in METRIC_NAMES}
    cumulative_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    candidates: list[str] = []
    references_text: list[str] = []
    for _, candidate, reference in pairs:
        candidates.append(candidate)
        references_text.append(reference)

    bert_rows: list[dict[str, float]] = []
    for offset in range(0, len(pairs), max(1, args.batch_size)):
        bert_rows.extend(
            bertscore_metrics(
                candidates[offset : offset + args.batch_size],
                references_text[offset : offset + args.batch_size],
                modules,
            )
        )

    for index, ((key, _, _), candidate, reference) in enumerate(
        zip(pairs, candidates, references_text), 1
    ):
        values = lexical_metrics(candidate, reference, modules)
        values.update(bert_rows[index - 1])
        values["blanc"] = (
            blanc_score(candidate, reference, modules) if not args.no_blanc else math.nan
        )
        totals = {
            name: total + values[name]
            for name, total in totals.items()
            if not (args.no_blanc and name == "blanc")
        } | ({"blanc": totals["blanc"]} if args.no_blanc else {})
        row: dict[str, Any] = {
            "index": index,
            "key": key,
            **values,
        }
        rows.append(row)
        cumulative = {
            "index": index,
            "key": row["key"],
            **{
                name: (
                    totals[name] / index
                    if not (args.no_blanc and name == "blanc")
                    else math.nan
                )
                for name in METRIC_NAMES
            },
        }
        cumulative_rows.append(cumulative)

    fieldnames = ["index", "key", *METRIC_NAMES]
    for path, output_rows in (
        (per_example_path, rows),
        (cumulative_path, cumulative_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    final = cumulative_rows[-1]
    summary = {
        "generated": str(args.generated),
        "ground_truth": str(args.ground_truth),
        "prediction_field": args.prediction_field,
        "reference_field": args.reference_field,
        "key_field": args.key_field,
        "matched_records": len(pairs),
        "skipped_generated_records": len(generated) - len(pairs),
        "unique_generated_keys": len(wanted),
        "metrics": {name: final[name] for name in METRIC_NAMES},
        "blanc_skipped": args.no_blanc,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
