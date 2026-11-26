"""Compare retrieval strategies on test cases 1-300.

Reads the retrieval reports of every strategy that has been run and writes
data/test_data/retrieval_comparison.{json,md}.

    python code/rag-setup/vdb/compare_strategies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdb_backends as vb  # noqa: E402

TD = vb.c.DATA_DIR / "test_data"
RUNS = [
    ("Vespa hybrid v1 (single query, top-5)", TD / "retrieval_vespa_v1" / "retrieval_report.json"),
    ("Vespa hybrid v2", TD / "retrieval_vespa_v2" / "retrieval_report.json"),
    ("Milvus dense baseline", TD / "retrieval_milvus" / "baseline" / "retrieval_report.json"),
    ("Milvus dense v2 pipeline", TD / "retrieval_milvus" / "retrieval_report.json"),
    ("Endee dense baseline", TD / "retrieval_endee" / "baseline" / "retrieval_report.json"),
    ("Endee dense v2 pipeline", TD / "retrieval_endee" / "retrieval_report.json"),
]


def metrics(report: dict) -> dict:
    if "metrics" in report:
        m = report["metrics"]
        return {t: {k: m[t].get(k) for k in ("recall@5", "recall@10", "recall@10_of_findable", "corpus_ceiling")}
                for t in ("statutes", "acts", "precedents")}
    r = report.get("recall_at_k", {})  # v1 layout: recall at top-5 only
    names = {"statutes": "section", "acts": "act", "precedents": "case"}
    return {t: {"recall@5": r.get(n, {}).get("recall"), "recall@10": None,
                "recall@10_of_findable": None, "corpus_ceiling": None} for t, n in names.items()}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main() -> int:
    rows = []
    for name, path in RUNS:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"strategy": name, "report": str(path.relative_to(vb.c.PROJECT_ROOT)),
                     "backend": report.get("backend"), "timings": report.get("timings"),
                     "metrics": metrics(report)})
    (TD / "retrieval_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Retrieval strategies on test cases 1-300",
        "",
        "Query: `Simplified_Facts`. Recall against the gold cited sections, Acts and precedents.",
        "",
        "| Strategy | Statutes R@5 | Statutes R@10 | Acts R@5 | Acts R@10 | Precedents R@5 | Precedents R@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(f"| {r['strategy']} | {pct(m['statutes']['recall@5'])} | {pct(m['statutes']['recall@10'])} | "
                     f"{pct(m['acts']['recall@5'])} | {pct(m['acts']['recall@10'])} | "
                     f"{pct(m['precedents']['recall@5'])} | {pct(m['precedents']['recall@10'])} |")
    lines += [
        "",
        "Baseline = one dense query over the whole facts, top-10 (the repo's Milvus / Endee scripts).",
        "v2 pipeline = the Vespa v2 pipeline (explicit references, per-passage and named-Act queries,",
        "citation propagation, weighted RRF) with the search swapped to the given engine; Milvus and",
        "Endee are dense-only, so the BM25 lists are absent. All strategies use the same",
        "arctic-embed-m-v2.0 vectors, self-exclusion and temporal cut-off.",
        "v1 counted Acts differently (660 gold Acts vs 462 later) and kept only the top 5.",
    ]
    for r in rows:
        if r.get("backend"):
            lines.append(f"\n**{r['strategy']}**: `{json.dumps(r['backend'])}`")
    (TD / "retrieval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
