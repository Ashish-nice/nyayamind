"""Ingest Central Acts (data/CA/Central_Acts.csv) into the `centralacts` schema.

Stages (see common.py):
    --stage embed   chunk (8192 tokens, 100 overlap) + embed -> data/vespa_feed/centralacts/
    --stage feed    push the pre-computed shards into Vespa
    --stage all     both (default)

The embedded text is "<statute> | Section <n>. <title> | <section text>" so that
a section whose body never names its Act can still be found by Act/topic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402


def make_docs(row_index: int, row: dict[str, str], tokenizer):
    section_text = c.preprocess_text(row.get("Section Text"), lower=False)
    if not section_text:
        return []
    statute = c.preprocess_text(row.get("Name of statute"), lower=False)
    number = c.preprocess_text(row.get("Section Number"), lower=False)
    title = c.preprocess_text(row.get("Section Title"), lower=False)
    header = f"{statute} | Section {number.rstrip('.')}. {title}"
    docs = []
    for chunk_index, chunk in enumerate(c.chunk_by_tokens(section_text, tokenizer, overlap=100)):
        doc_id = f"ca_{row_index}_{chunk_index}"
        fields = {
            "id": doc_id,
            "name_of_statute": statute,
            "section_number": number,
            "section_title": title,
            "section_text": chunk,
        }
        docs.append((doc_id, fields, f"{header} | {chunk}"))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_stage_args(parser)
    args = parser.parse_args()
    if args.stage in ("embed", "all"):
        c.run_embedding_stage(
            c.CA_SCHEMA, enumerate(c.iter_csv(c.CA_CSV)), make_docs, total=36731, limit=args.limit
        )
    if args.stage in ("feed", "all"):
        c.run_feed_stage(c.CA_SCHEMA, follow=args.follow)


if __name__ == "__main__":
    main()
