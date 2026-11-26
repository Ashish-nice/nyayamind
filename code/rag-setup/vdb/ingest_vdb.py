"""Load the pre-computed corpus embeddings into Milvus or Endee.

Reads the offline shards in data/vespa_feed (same embeddings as the Vespa
collections) and writes four collections: centralacts, stateacts, scjudgments,
hcjudgments. On the first run it also builds the SQLite judgment-text side
store (data/vdb_textstore.sqlite) used by the retrieval script.

    python code/rag-setup/vdb/ingest_vdb.py --backend milvus
    python code/rag-setup/vdb/ingest_vdb.py --backend endee

A collection whose row count already matches the shards is skipped, so the
command can be re-run after an interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vdb_backends as vb  # noqa: E402


def expected_docs(collection: str) -> int:
    total = 0
    for schema in vb.COLLECTIONS[collection][1]:
        progress = vb.c.FEED_DIR / schema / "_progress.json"
        if progress.exists():
            total += int(json.loads(progress.read_text()).get("docs", 0))
    return total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("milvus", "endee"), required=True)
    p.add_argument("--collections", nargs="*", default=list(vb.COLLECTIONS))
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--limit-docs", type=int, default=None, help="smoke test: stop after N docs per collection")
    p.add_argument("--no-textstore", action="store_true")
    p.add_argument("--force", action="store_true", help="recreate collections even if complete")
    args = p.parse_args()

    backend = vb.make_backend(args.backend)
    for _ in range(60):
        if backend.ready():
            break
        time.sleep(5)
    else:
        raise SystemExit(f"{args.backend} server is not reachable")

    build_store = not args.no_textstore and not vb.TextStore.exists() and args.limit_docs is None
    writer = vb.TextStoreWriter() if build_store else None
    timings = {}
    for collection in args.collections:
        kind = vb.COLLECTIONS[collection][0]
        want = expected_docs(collection)
        have = backend.count(collection)
        needs_text = writer is not None and kind == "case"
        if not args.force and args.limit_docs is None and want and have == want and not needs_text:
            print(f"[{collection}] complete ({have} docs) - skipping", flush=True)
            backend.load(collection)
            continue
        started = time.time()
        print(f"[{collection}] ingesting {want} docs into {args.backend} ...", flush=True)
        backend.create(collection)
        docs, vecs, n = [], [], 0
        for doc in vb.iter_docs(collection):
            f = doc["fields"]
            docs.append(doc)
            vecs.append(vb.decode_bf16(f["embedding"]["values"]))
            if needs_text:
                writer.add(doc["id"], str(f.get("content") or f.get("text") or ""))
            n += 1
            if len(docs) >= args.batch:
                backend.insert(collection, docs, np.stack(vecs))
                docs, vecs = [], []
                if n % 50000 < args.batch:
                    rate = n / (time.time() - started)
                    print(f"  [{collection}] {n}/{want} ({rate:.0f} docs/s)", flush=True)
            if args.limit_docs and n >= args.limit_docs:
                break
        if docs:
            backend.insert(collection, docs, np.stack(vecs))
        print(f"  [{collection}] inserted {n}; building index / loading ...", flush=True)
        backend.finalize(collection)
        timings[collection] = round(time.time() - started, 1)
        print(f"[{collection}] done in {timings[collection]}s", flush=True)
    if writer is not None:
        writer.close()
        print(f"text store written: {vb.TEXT_STORE}", flush=True)
    out = vb.c.DATA_DIR / "test_data" / f"retrieval_{args.backend}"
    out.mkdir(parents=True, exist_ok=True)
    if args.limit_docs is None:
        (out / "ingest_timings.json").write_text(json.dumps(
            {"backend": backend.describe(), "seconds": timings,
             "counts": {k: backend.count(k) for k in args.collections}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
