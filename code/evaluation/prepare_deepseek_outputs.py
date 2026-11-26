"""Turn fine-tuned DeepSeek-R1 inference files into evaluator-ready records.

The merged DeepSeek files keep the model's text only in ``finetuned_model_output``.
Their ``Arguments of Petitioner`` / ``Arguments of Respondent`` columns are the
dataset's *gold* columns, so evaluating the files as-is would score gold text.
This script parses the model output into the fields the evaluators read and
overwrites those columns with model text (blank when the model did not write
the section; nothing is ever filled from ground truth).

It also sets the statute/precedent fields to the context the model actually
saw, because the LLM judge scores faithfulness against them:

    --context gold   keep the record's gold section_titles / cited_cases
    --context rag    copy the retrieved context from test_v2_rag2.json (by case)
    --context none   empty lists (facts-only run)

    python code/evaluation/prepare_deepseek_outputs.py \
        --input logs/deepseek_rag2_context_merged.json --context rag \
        --output logs/deepseek_rag2_context_full_run.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "inference"))
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
from rag_inference_common import parse_explanation, parse_sections  # noqa: E402
from evaluate_text_metrics import decision_label  # noqa: E402

CONTEXT_FIELDS = ("section_titles", "section_texts", "cited_cases", "cited_case_ids",
                  "cited_case_judgments")


def key_of(record: dict) -> str:
    return str(record.get("document_id") or "").strip() or str(record.get("Case Name", "")).strip()


def prediction_text(output: str) -> str:
    m = re.search(r"#{1,3}\s*PREDICTION\s*:?\s*(.*?)(?=\n?\s*#{1,3}\s*EXPLANATION|$)",
                  output, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", choices=("gold", "rag", "none"), required=True)
    p.add_argument("--rag-dataset", type=Path, default=ROOT / "data/test_data/test_v2_rag2.json")
    p.add_argument("--output-field", default="finetuned_model_output")
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) else data
    rag = None
    if args.context == "rag":
        rag_rows = json.loads(args.rag_dataset.read_text(encoding="utf-8"))
        if [key_of(r) for r in rows[: len(rag_rows)]] != [key_of(r) for r in rag_rows[: len(rows)]]:
            raise SystemExit("record order differs from the RAG dataset; cannot align contexts by case")
        rag = rag_rows

    results, missing, labels = [], Counter(), Counter()
    for index, row in enumerate(rows):
        raw = str(row.get(args.output_field) or "")
        text = raw.replace("：", ":")          # full-width colon used by the model
        sections = parse_sections(text)
        explanation = parse_explanation(text)
        pred_text = prediction_text(text)
        label = decision_label(pred_text) if pred_text else ""
        prediction = int(label) if label in ("0", "1") else None
        record = dict(row)
        record.pop("prompt", None)
        record.update({
            "case_index": index,
            "Legal Issue Analysis": sections["Legal Issue Analysis"],
            "Arguments of Petitioner": sections["Arguments of Petitioner"],
            "Arguments of Respondent": sections["Arguments of Respondent"],
            "Deliberation": sections["Deliberation"],
            "Explanation": explanation,
            "Prediction": prediction,
            "Prediction_text": pred_text,
            "predicted_decision": prediction,
        })
        if args.context == "rag":
            for field in CONTEXT_FIELDS:
                record[field] = rag[index].get(field) or []
        elif args.context == "none":
            for field in CONTEXT_FIELDS:
                record[field] = []
        record["missing_fields"] = [
            f for f in ("Legal Issue Analysis", "Arguments of Petitioner", "Arguments of Respondent",
                        "Deliberation", "Explanation") if not record[f]
        ] + (["Prediction"] if prediction is None else [])
        missing.update(record["missing_fields"])
        labels[prediction] += 1
        results.append(record)

    envelope = {
        "run_name": args.output.stem,
        "model_path": data.get("model_path", "DeepSeek-R1-Distill-Qwen-14B (fine-tuned)") if isinstance(data, dict) else "",
        "source_file": str(args.input),
        "context": args.context,
        "gold_fallback_disabled": True,
        "summary": {"n_cases": len(results), "missing_by_field": dict(missing),
                    "prediction_distribution": {str(k): v for k, v in labels.items()}},
        "results": results,
    }
    args.output.write_text(json.dumps(envelope, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(envelope["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
