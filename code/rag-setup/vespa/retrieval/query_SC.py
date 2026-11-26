"""Hybrid (dense + BM25, RRF) search over Supreme Court judgments (`scjudgments`).

Results are de-duplicated per judgment (best-ranked chunk kept).

Endpoint for other code:
    from query_SC import search_sc_judgments
    search_sc_judgments("dowry death presumption", top_k=5)

Interactive use:
    python code/rag-setup/vespa/retrieval/query_SC.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

CSV_FILENAME = "sc_search_results.csv"


def search_sc_judgments(query_text: str, top_k: int = 5, query_embedding=None, exclude=None, app=None) -> list[dict]:
    return c.search_collection(
        [c.SC_SCHEMA],
        query_text,
        top_k=top_k,
        query_embedding=query_embedding,
        dedupe_fields=("diary_number", "case_name", "judgment_date"),
        exclude=exclude,
        app=app,
    )


def save_to_csv(query: str, results: list[dict]) -> None:
    fieldnames = ["Query", "Rank", "ID", "Case Name", "Date", "Text Snippet"]
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
                    "ID": r["vespa_id"],
                    "Case Name": r.get("case_name"),
                    "Date": r.get("judgment_date"),
                    "Text Snippet": r.get("content", "")[:500],
                }
            )
    print(f" >> Saved {len(results)} rows to {CSV_FILENAME}")


if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == "exit":
                break
            results = search_sc_judgments(user_query)
            print("\n" + "=" * 60 + f"\nTop {len(results)} Results for: '{user_query}'\n" + "=" * 60)
            for r in results:
                print(f"\n{r['rank']}. CASE: {r.get('case_name')} ({r.get('judgment_date')})")
                print(f"   ID: {r['vespa_id']} | rrf={r['rrf_score']}")
            save_to_csv(user_query, results)
    except KeyboardInterrupt:
        print("\nExiting...")
