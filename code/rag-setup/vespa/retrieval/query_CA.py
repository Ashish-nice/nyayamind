"""Hybrid (dense + BM25, RRF) search over Central Acts (`centralacts`).

Endpoint for other code:
    from query_CA import search_central_acts
    search_central_acts("bail in non-bailable offence", top_k=5)

Interactive use:
    python code/rag-setup/vespa/retrieval/query_CA.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

CSV_FILENAME = "ca_search_results.csv"


def search_central_acts(query_text: str, top_k: int = 5, query_embedding=None, app=None) -> list[dict]:
    return c.search_collection(
        [c.CA_SCHEMA],
        query_text,
        top_k=top_k,
        query_embedding=query_embedding,
        dedupe_fields=("name_of_statute", "section_number"),
        app=app,
    )


def save_to_csv(query: str, results: list[dict]) -> None:
    fieldnames = ["Query", "Rank", "Act Name", "Section Number", "Section Title", "Snippet"]
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
                    "Act Name": r.get("name_of_statute"),
                    "Section Number": r.get("section_number"),
                    "Section Title": r.get("section_title"),
                    "Snippet": r.get("section_text", "")[:1000],
                }
            )
    print(f" >> Saved {len(results)} rows to {CSV_FILENAME}")


if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == "exit":
                break
            results = search_central_acts(user_query)
            print("\n--- Top Central Acts Results ---")
            for r in results:
                print(f"{r['rank']}. {c.statute_title(r)}  (rrf={r['rrf_score']})")
                print(f"   Title: {r.get('section_title')}")
                print(f"   Snippet: {r.get('section_text', '')[:200]}...\n")
            save_to_csv(user_query, results)
    except KeyboardInterrupt:
        print("\nExiting...")
