"""Ingest State Acts (data/SA/<State>.csv) into one `sa_<state>` schema per state.

Stages (see common.py):
    --stage embed   chunk (8192 tokens, 200 overlap) + embed -> data/vespa_feed/sa_<state>/
    --stage feed    push the pre-computed shards into Vespa
    --stage all     both (default)

    --states "Goa" "Kerala"   restrict to some states (default: every CSV in data/SA)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402


def make_docs_for(state_name: str, schema_name: str):
    def make_docs(row_index: int, row: dict[str, str], tokenizer):
        section_text = c.preprocess_text(row.get("Section Text"), lower=False)
        if not section_text:
            return []
        # State CSVs use underscores in statute names ("The_Goa_..._Act,_2005").
        statute = c.preprocess_text(str(row.get("Name of Statute") or "").replace("_", " "), lower=False)
        number = c.preprocess_text(row.get("Section Number"), lower=False)
        title = c.preprocess_text(row.get("Section Title"), lower=False)
        raw_id = str(row.get("id") or "").strip()
        original_id = int(raw_id) if raw_id.isdigit() else row_index
        header = f"{statute} ({state_name}) | Section {number.rstrip('.')}. {title}"
        docs = []
        for chunk_index, chunk in enumerate(c.chunk_by_tokens(section_text, tokenizer, overlap=200)):
            doc_id = f"{schema_name}_{original_id}_{chunk_index}"
            fields = {
                "id": doc_id,
                "original_doc_id": original_id,
                "state_name": c.preprocess_text(row.get("State Name") or state_name, lower=False),
                "name_of_statute": statute,
                "section_number": number,
                "section_title": title,
                "section_text": chunk,
            }
            docs.append((doc_id, fields, f"{header} | {chunk}"))
        return docs

    return make_docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_stage_args(parser)
    parser.add_argument("--states", nargs="*", default=None)
    args = parser.parse_args()
    states = args.states or c.available_states()
    for state_name in states:
        if not c.state_csv(state_name).exists():
            print(f"Skipping {state_name}: no CSV at {c.state_csv(state_name)}")
            continue
        schema_name = c.state_schema(state_name)
        if args.stage in ("embed", "all"):
            c.run_embedding_stage(
                schema_name,
                enumerate(c.iter_csv(c.state_csv(state_name))),
                make_docs_for(state_name, schema_name),
                limit=args.limit,
            )
        if args.stage in ("feed", "all"):
            c.run_feed_stage(schema_name, follow=args.follow)


if __name__ == "__main__":
    main()
