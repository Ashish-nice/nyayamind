"""Blind LLM-as-a-judge scoring with Claude Haiku 5.5: whole analysis + each section.

Runs evaluate_llm_judge.py once per (run, section). "overall" judges the whole generated
analysis against the court's reasoning (written to <run>/llm_judge); each other section is
judged alone against its own simplified gold field (<run>/llm_judge_sections/<section>). Prediction is not judged (it is a 0/1 label; decision
accuracy is reported instead).

Runs: OpenRouter models with retrieved citations (cases 1-300), the earlier gold-citation
runs (leak-free records only: sections copied from ground truth are excluded), and the
three DeepSeek context conditions.

Outputs: evaluation_results/<run>/llm_judge_sections/<section>/{summary.json,...}
The judge caches per case, so an interrupted run resumes where it stopped.

    python code/evaluation/judge_sections.py --dry-run     # alignment counts, no API calls
    python code/evaluation/judge_sections.py               # judge everything
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
from evaluate_text_metrics import record_key  # noqa: E402

GT = ROOT / "data/test_data/test_v2.json"
ER = ROOT / "evaluation_results"
WORK = ROOT / "logs/judge_sections"

SECTIONS = {  # section -> (generated field, gold field)
    "legal_issue": ("Legal Issue Analysis", "Simplified_Issue"),
    "petitioner": ("Arguments of Petitioner", "Simplified_Arguments_of_Petitioner"),
    "respondent": ("Arguments of Respondent", "Simplified_Arguments_of_Respondent"),
    "explanation": ("Explanation", "Simplified_Reasoning"),
}
JOBS = {  # job -> (generated field, gold field, prediction label, reference label)
    "overall": ("combined_reasoning", "Simplified_Reasoning",
                "Generated Legal Analysis", "Actual Court Reasoning"),
    "legal_issue": (*SECTIONS["legal_issue"], "Generated Legal Issue Analysis",
                    "Actual Legal Issues (as framed in the judgment)"),
    "petitioner": (*SECTIONS["petitioner"], "Generated Arguments of the Petitioner",
                   "Actual Arguments of the Petitioner (as recorded in the judgment)"),
    "respondent": (*SECTIONS["respondent"], "Generated Arguments of the Respondent",
                   "Actual Arguments of the Respondent (as recorded in the judgment)"),
    "explanation": (*SECTIONS["explanation"], "Generated Explanation", "Actual Court Reasoning"),
}

# (result dir, generated file, limit, leak-free filtering)
RUNS = [
    ("nemotron_3_ultra_300", "logs/nemotron_3_ultra_550b_712.json", 300, True),
    ("nemotron_3_ultra_full_run", "logs/nemotron_3_ultra_full_run.json", None, False),
    ("opus_5_5_100", "logs/claude_opus_5_5_fixed.json", None, True),
    ("claude_opus_5_5_full_run", "logs/claude_opus_5_5_full_run.json", None, False),
    ("gpt6_1_sol_300", "logs/gpt6_1_sol_350.json", 300, True),
    ("gpt_6_1_sol_full_run", "logs/gpt_6_1_sol_full_run.json", None, False),
    ("deepseek_no_citations_full_run", "logs/deepseek_no_citations_full_run.json", None, False),
    ("deepseek_groundtruth_context_full_run", "logs/deepseek_groundtruth_context_full_run.json", None, False),
    ("deepseek_rag2_context_full_run", "logs/deepseek_rag2_context_full_run.json", None, False),
]


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def load(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["results"] if isinstance(d, dict) else d


def copied_keys(records: list[dict]) -> set[str]:
    """Keys of records with any section copied verbatim from the gold text. Checked record by
    record against every gold record sharing the key, so duplicated IDs cannot hide a copy."""
    gold: dict[str, list[dict]] = {}
    for g in load(GT):
        gold.setdefault(record_key(g, "document_id"), []).append(g)
    out = set()
    for r in records:
        k = record_key(r, "document_id")
        for g in gold.get(k, []):
            if any(norm(r.get(a)) and norm(r.get(a)) == norm(g.get(b)) for a, b in SECTIONS.values()):
                out.add(k)
    return out


def leak_free(records: list[dict]) -> list[dict]:
    """Drop every record whose key has a copied section (the same rule build_summaries.py
    applies to the text metrics, so text and judge scores use one subset)."""
    bad = copied_keys(records)
    return [r for r in records if record_key(r, "document_id") not in bad]


def prepare(run: str, src: str, limit: int | None, clean: bool) -> tuple[Path, list[dict]]:
    records = load(ROOT / src)
    if limit:
        records = records[:limit]
    if not clean:
        return ROOT / src, records
    records = leak_free(records)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{run}_leakfree.json"
    path.write_text(json.dumps({"results": records}, ensure_ascii=False), encoding="utf-8")
    return path, records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--runs", nargs="*", help="subset of result dirs")
    p.add_argument("--sections", nargs="*", default=list(JOBS))
    a = p.parse_args()

    for run, src, limit, clean in RUNS:
        if a.runs and run not in a.runs:
            continue
        gen, records = prepare(run, src, limit, clean)
        for sec in a.sections:
            field, ref, pred_label, ref_label = JOBS[sec]
            n = (len(records) if sec == "overall" else sum(1 for r in records if norm(r.get(field))))
            out = ER / run / ("llm_judge" if sec == "overall" else f"llm_judge_sections/{sec}")
            if a.dry_run:
                print(f"{run:40s} {sec:12s} records={len(records):3d} non-blank={n:3d} -> {out.relative_to(ROOT)}")
                continue
            if (out / "summary.json").exists():
                print(f"skip {run}/{sec} (done)", flush=True)
                continue
            print(f"=== {run} / {sec} ({n} cases)", flush=True)
            cmd = [sys.executable, "-X", "utf8", str(ROOT / "code/evaluation/evaluate_llm_judge.py"),
                   "--generated", str(gen), "--ground-truth", str(GT), "--output-dir", str(out),
                   "--prediction-field", field, "--reference-field", ref, "--workers", str(a.workers),
                   "--pred-label", pred_label, "--ref-label", ref_label]
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"ABORTED at {run}/{sec} (exit {rc})", flush=True)
                return rc
    print("all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
