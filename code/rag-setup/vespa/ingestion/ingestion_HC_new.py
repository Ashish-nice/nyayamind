"""Ingest High Court judgments (data/HC/hc_cases_judgements.csv) into `hcjudgments`.

The 9 GB CSV is streamed row by row (never loaded into pandas).

Default (--max-chunks 1) reproduces the paper's Vespa HC collection, which holds
one document per judgment (811,492 docs): the embedding covers the first
8192 tokens and BM25 indexes the first 60,000 characters of the judgment.
Use --max-chunks 0 to index every 8192-token chunk instead (about 3x the GPU
time). --embed-tokens trades recall for speed on a laptop GPU: e.g. 2048
embeds only the head of each judgment (parties, facts, issues) and roughly
halves the embedding time.

Stages (see common.py):
    --stage embed   chunk + embed -> data/vespa_feed/hcjudgments/
    --stage feed    push the pre-computed shards into Vespa
    --stage all     both (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

HC_ROWS = 811493
BM25_CHARS = 60000


def make_docs_factory(max_chunks: int, embed_tokens: int):
    def make_docs(row_index: int, row: dict[str, str], tokenizer):
        raw = row.get("Text") or ""
        if max_chunks:
            # Only the head is embedded; avoid tokenising the whole judgment.
            raw = raw[: max(BM25_CHARS, embed_tokens * max_chunks * 8)]
        text = c.preprocess_text(raw, lower=False)
        if not text:
            return []
        metadata = {
            "case_name": c.preprocess_text(row.get("Title"), lower=False)[:2000],
            "court_name": c.preprocess_text(row.get("Court Name"), lower=False),
            "judgment_date": c.preprocess_text(row.get("Decision Date"), lower=False),
            "citations": c.preprocess_text(row.get("Case Number"), lower=False),
            "bench": c.preprocess_text(row.get("Judge"), lower=False)[:2000],
        }
        if max_chunks == 1:  # one doc per judgment: only the head is tokenised
            chunks = [c.head_by_tokens(text, tokenizer, embed_tokens)]
        else:
            chunks = c.chunk_by_tokens(
                text, tokenizer, chunk_tokens=embed_tokens, overlap=100, max_chunks=max_chunks or None
            )
        docs = []
        for chunk_index, chunk in enumerate(chunks):
            doc_id = f"hc_{row_index}_{chunk_index}"
            stored = text[:BM25_CHARS] if max_chunks == 1 else chunk
            fields = {"id": doc_id, "chunk_index": chunk_index, "text": stored, **metadata}
            docs.append((doc_id, fields, chunk))
        return docs

    return make_docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_stage_args(parser)
    parser.add_argument("--max-chunks", type=int, default=1, help="Chunks per judgment; 0 = all (default 1).")
    args = parser.parse_args()
    if args.stage in ("embed", "all"):
        c.run_embedding_stage(
            c.HC_SCHEMA,
            enumerate(c.iter_csv(c.HC_CSV)),
            make_docs_factory(args.max_chunks, args.embed_tokens),
            total=HC_ROWS,
            limit=args.limit,
            docs_per_batch=512,
        )
    if args.stage in ("feed", "all"):
        c.run_feed_stage(c.HC_SCHEMA, follow=args.follow)


if __name__ == "__main__":
    main()
