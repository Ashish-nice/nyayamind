"""Ingest Supreme Court judgments (data/SC/SC_data.csv) into `scjudgments`.

Each judgment is split into chunks of up to 8192 tokens (100-token overlap);
every chunk becomes one Vespa document carrying the judgment's metadata.

Stages (see common.py):
    --stage embed   chunk + embed -> data/vespa_feed/scjudgments/
    --stage feed    push the pre-computed shards into Vespa
    --stage all     both (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

METADATA = {  # Vespa field -> CSV column
    "diary_number": "diary_no",
    "judgment_type": "Judgement_type",
    "case_number": "case_no",
    "petitioner": "pet",
    "respondent": "res",
    "petitioner_advocate": "pet_adv",
    "respondent_advocate": "res_adv",
    "bench": "bench",
    "judgment_by": "judgement_by",
    "judgment_date": "judgment_dates",
    "temp_link": "temp_link",
    "citations": "citation",
}


EMBED_TOKENS = c.CHUNK_TOKENS  # set from --embed-tokens


def make_docs(row_index: int, row: dict[str, str], tokenizer):
    content = c.preprocess_text(row.get("content"), lower=False)
    if not content:
        return []
    case_name = c.preprocess_text(row.get("file_name"), lower=False)
    if not case_name:
        case_name = f"{c.preprocess_text(row.get('pet'), False)} v. {c.preprocess_text(row.get('res'), False)}"
    metadata = {field: c.preprocess_text(row.get(column), lower=False) for field, column in METADATA.items()}
    source_id = str(row.get("id") or row_index).strip()
    docs = []
    for chunk_index, chunk in enumerate(c.chunk_by_tokens(content, tokenizer, overlap=100)):
        doc_id = f"sc_{source_id}_{chunk_index}"
        fields = {
            "id": doc_id,
            "case_name": case_name,
            "chunk_index": chunk_index,
            "content": chunk,
            **metadata,
        }
        embed_text = chunk if EMBED_TOKENS >= c.CHUNK_TOKENS else c.head_by_tokens(chunk, tokenizer, EMBED_TOKENS)
        docs.append((doc_id, fields, embed_text))
    return docs


def main() -> None:
    global EMBED_TOKENS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_stage_args(parser)
    args = parser.parse_args()
    EMBED_TOKENS = args.embed_tokens
    if args.stage in ("embed", "all"):
        c.run_embedding_stage(
            c.SC_SCHEMA, enumerate(c.iter_csv(c.SC_CSV)), make_docs, total=37398, limit=args.limit,
            docs_per_batch=128,
        )
    if args.stage in ("feed", "all"):
        c.run_feed_stage(c.SC_SCHEMA, follow=args.follow)


if __name__ == "__main__":
    main()
