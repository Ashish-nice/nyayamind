"""Hybrid (dense + BM25, RRF) search over High Court judgments (`hcjudgments`).

Results are de-duplicated per judgment (best-ranked chunk kept).

Endpoint for other code:
    from query_HC_new import search_hc_judgments
    search_hc_judgments("anticipatory bail in matrimonial dispute", top_k=5)

Interactive use:
    python code/rag-setup/vespa/retrieval/query_HC_new.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

CSV_FILENAME = "hc_search_results_multi.csv"


def search_hc_judgments(query_text: str, top_k: int = 5, query_embedding=None, exclude=None, app=None) -> list[dict]:
    return c.search_collection(
        [c.HC_SCHEMA],
        query_text,
        top_k=top_k,
        query_embedding=query_embedding,
        dedupe_fields=("case_name", "citations", "judgment_date"),
        exclude=exclude,
        app=app,
    )


def save_to_csv(query: str, results: list[dict]) -> None:
    fieldnames = ["Query", "Rank", "Score", "Source", "Case Name", "Bench", "Full Text"]
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
                    "Score": r["rrf_score"],
                    "Source": r.get("court_name"),
                    "Case Name": r.get("case_name"),
                    "Bench": r.get("bench"),
                    "Full Text": r.get("text", ""),
                }
            )
    print(f" >> Results appended to {CSV_FILENAME}")


if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == "exit":
                break
            results = search_hc_judgments(user_query)
            print("\n--- Top Results (RRF) ---")
            for r in results:
                print(f"Rank {r['rank']} | Score: {r['rrf_score']:.4f} | {r.get('court_name')}")
                print(f"Case: {r.get('case_name')}\nBench: {r.get('bench')}")
                print(f"Snippet: {r.get('text', '')[:100]}...\n" + "-" * 50)
            save_to_csv(user_query, results)
    except KeyboardInterrupt:
        print("\nExiting...")
