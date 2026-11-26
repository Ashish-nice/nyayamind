"""Test-set retrieval (cases 1-300) with the Milvus or Endee strategy.

Reuses the Vespa v2 retrieval pipeline (code/rag-setup/vespa/retrieval/
retrieve_test_set.py: query = Simplified_Facts, self-exclusion, temporal
cut-off, explicit statute references, per-passage queries, named-Act search,
citation propagation, weighted RRF, same output layout and evaluation) and
swaps only the search backend. Milvus and Endee are dense-only strategies, so
the BM25 evidence lists are empty and their weights drop out.

Two outputs per backend, both on cases 1-300:

    data/test_data/retrieval_<backend>/            v2 pipeline on the backend
        test_v2_rag.json, test_v2_rag_inference.json, retrieval_report.json
    data/test_data/retrieval_<backend>/baseline/   strategy as in the repo scripts:
        one dense query over the whole facts, top-10 statutes + top-10 precedents

The flat v2 file is also copied to data/test_data/test_v2_rag_<backend>.json.
Query embeddings are reused from data/test_data/retrieval_vespa_v2 (same
model, same passages), so no GPU is needed.

    python code/rag-setup/vdb/retrieve_test_set_vdb.py --backend milvus
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vespa" / "retrieval"))
import vdb_backends as vb  # noqa: E402
import retrieve_test_set as rts  # noqa: E402

c = vb.c
BASELINE_WEIGHTS = {"statutes": {"facts_dense": 1.0}, "precedents": {"facts_dense": 1.0}}


def install_backend(backend, store: vb.TextStore) -> None:
    """Point the Vespa pipeline's search / text functions at the backend."""

    def search_kind(kind: str, vector, hits: int, acts):
        merged = []
        for collection in vb.KIND_COLLECTIONS[kind]:
            merged.extend(backend.search(collection, vector, hits, acts))
        merged.sort(key=lambda h: -h["score"])
        merged = merged[:hits]
        if kind == "case":
            bodies = store.get_many([h["id"] for h in merged])
            for h in merged:
                h["fields"]["content"] = bodies.get(h["id"], "")
        return merged

    def hybrid(app, sources, text, vector, hits, where="", bm25=True):  # noqa: ARG001
        kind = "case" if c.SC_SCHEMA in sources else "statute"
        acts = json.loads(where) if where else None
        return search_kind(kind, vector, hits, acts), []

    def act_filter(catalog, act):
        names = sorted({catalog.entries[i]["name_of_statute"] for i in catalog.acts[act]})
        return json.dumps(names) if names else ""

    def fetch_texts(app, ids, cap):  # noqa: ARG001
        return {k: v[:cap] for k, v in store.get_many(list(ids)).items()}

    rts.hybrid = hybrid
    rts.act_filter = act_filter
    rts.fetch_texts = fetch_texts
    c.vespa_app = lambda: backend


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("milvus", "endee"), required=True)
    p.add_argument("--stage", choices=("retrieve", "merge", "all"), default="all")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out_dir = c.DATA_DIR / "test_data" / f"retrieval_{a.backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = c.DATA_DIR / "test_data" / "retrieval_vespa_v2"
    for name in ("queries.json", "query_embeddings.npy"):
        if not (out_dir / name).exists():
            shutil.copy2(src / name, out_dir / name)

    if not vb.TextStore.exists():
        raise SystemExit("text store missing - run ingest_vdb.py first")
    backend = vb.make_backend(a.backend)
    if not backend.ready():
        raise SystemExit(f"{a.backend} server not reachable")
    for collection in vb.COLLECTIONS:
        backend.load(collection)
    install_backend(backend, vb.TextStore())

    # same settings as the Vespa v2 run
    args = argparse.Namespace(
        dataset=c.TEST_JSON, out_dir=out_dir, start=0, limit=a.limit, top_k=10, hits=30,
        case_hits=30, passage_hits=10, propagate_from=30, workers=a.workers,
        max_text_chars=8000, weights=None, overwrite=a.overwrite,
    )
    timings = {}
    if a.stage in ("retrieve", "all"):
        started = time.time()
        rts.stage_retrieve(args)
        timings["retrieve_seconds"] = round(time.time() - started, 1)

    if a.stage in ("merge", "all"):
        # 1) v2 pipeline on this backend (dense-only lists; BM25 weights drop out)
        rts.STRATEGY = f"{a.backend}-dense-v2"
        started = time.time()
        rts.stage_merge(args)
        timings["merge_seconds"] = round(time.time() - started, 1)
        shutil.copy2(out_dir / "test_v2_rag_inference.json",
                     c.DATA_DIR / "test_data" / f"test_v2_rag_{a.backend}.json")

        # 2) baseline: single whole-facts dense query, top-10 (repo strategy)
        base = out_dir / "baseline"
        base.mkdir(exist_ok=True)
        shutil.copy2(out_dir / "candidates.jsonl", base / "candidates.jsonl")
        weights = base / "weights.json"
        weights.write_text(json.dumps(BASELINE_WEIGHTS, indent=2))
        rts.STRATEGY = f"{a.backend}-dense-baseline"
        rts.stage_merge(argparse.Namespace(**{**vars(args), "out_dir": base, "weights": weights}))

    report = out_dir / "retrieval_report.json"
    if report.exists():
        data = json.loads(report.read_text())
        data["backend"] = backend.describe()
        data["timings"] = {**data.get("timings", {}), **timings}
        report.write_text(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
