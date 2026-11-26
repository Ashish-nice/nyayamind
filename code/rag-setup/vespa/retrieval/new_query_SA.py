"""Hybrid (dense + BM25, RRF) search over State Acts (`sa_<state>` schemas).

All states are searched together (one query over every state schema), as in
the paper; pass ``states=[...]`` to restrict.

Endpoint for other code:
    from new_query_SA import search_state_acts
    search_state_acts("rent control eviction", top_k=5, states=["Maharashtra"])

Interactive use:
    python code/rag-setup/vespa/retrieval/new_query_SA.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

CSV_FILENAME = "sa_search_results_multi.csv"


def search_state_acts(
    query_text: str, top_k: int = 5, states: list[str] | None = None, query_embedding=None, app=None
) -> list[dict]:
    sources = [c.state_schema(s) for s in states] if states else c.sa_schemas()
    return c.search_collection(
        sources,
        query_text,
        top_k=top_k,
        query_embedding=query_embedding,
        dedupe_fields=("state_name", "name_of_statute", "section_number"),
        app=app,
    )


def save_to_csv(query: str, results: list[dict]) -> None:
    fieldnames = ["Query", "Rank", "State", "Statute", "Section Number", "Section Title", "Section Text Snippet", "source"]
    exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Query": query,
                    "Rank": r["rank"],
                    "State": r.get("state_name"),
                    "Statute": r.get("name_of_statute"),
                    "Section Number": r.get("section_number"),
                    "Section Title": r.get("section_title"),
                    "Section Text Snippet": r.get("section_text", "")[:1000],
                    "source": r["schema"],
                }
            )
    print(f" >> Saved {len(results)} rows to {CSV_FILENAME}")


if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == "exit":
                break
            results = search_state_acts(user_query)
            print("\nTop State Act Results:\n" + "-" * 50)
            for r in results:
                print(f"{r['rank']}. [{r.get('state_name')}] {c.statute_title(r)}")
                print(f"   {r.get('section_title')} | source: {r['schema']} | rrf={r['rrf_score']}")
                print(f"   Snippet: {r.get('section_text', '')[:150]}...\n" + "-" * 50)
            save_to_csv(user_query, results)
    except KeyboardInterrupt:
        print("\nExiting...")
