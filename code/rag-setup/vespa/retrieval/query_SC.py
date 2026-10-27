import sys
import json
from vespa.application import Vespa
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch
import csv
import os

# --- Configuration ---
VESPA_URL = "http://localhost:8083"
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
RRF_K = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_FILENAME = "sc_search_results.csv"

def get_vespa_app():
    return Vespa(url=VESPA_URL)

def reciprocal_rank_fusion(results_dict: Dict[str, List[Dict]], k=RRF_K):
    rrf_score = {}

    for method, hits in results_dict.items():
        for rank, hit in enumerate(hits):
            doc_id = hit.get('id')
            if doc_id not in rrf_score:
                rrf_score[doc_id] = {"score": 0, "hit": hit}
            rrf_score[doc_id]["score"] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(rrf_score.values(), key=lambda x: x['score'], reverse=True)
    return [item['hit'] for item in sorted_docs]

def save_to_csv(query, results):
    """Helper function to save results to CSV."""
    file_exists = os.path.isfile(CSV_FILENAME)
    
    csv_rows = []
    for i, hit in enumerate(results):
        fields = hit.get('fields', {})
        csv_rows.append({
            "Query": query,
            "Rank": i + 1,
            "ID": hit.get('id'),
            "Case Name": fields.get('case_name', 'Unknown'),
            "Date": fields.get('date', 'N/A'),
            # Saving a snippet of text (first 500 chars) to keep CSV manageable
            "Text Snippet": fields.get('content', '')[:500] 
        })

    if not csv_rows:
        return

    try:
        with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Query", "Rank", "ID", "Case Name", "Date", "Text Snippet"])
            
            # Write header only if file didn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(csv_rows)
            print(f" >> Saved {len(csv_rows)} rows to {CSV_FILENAME}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def search_sc_judgments(query_text, top_k=5):
    app = get_vespa_app()
    print(f"Encoding query: '{query_text}'")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
    query_emb = model.encode(query_text).tolist()

    # --- 1. Vector Search ---
    vector_res = app.query(body={
        "yql": "select * from scjudgments where {targetHits:10}nearestNeighbor(embedding, q_emb)",
        "ranking.profile": "semantic",
        "input.query(q_emb)": query_emb,
        "hits": 10
    })

    # --- 2. BM25 Search ---
    bm25_res = app.query(body={
        "yql": "select * from scjudgments where userQuery()",
        "query": query_text,
        "ranking.profile": "bm25",
        "hits": 10
    })

    hits_vector = vector_res.hits if vector_res.hits else []
    hits_bm25 = bm25_res.hits if bm25_res.hits else []

    print(f"   -> Found {len(hits_vector)} vector hits and {len(hits_bm25)} BM25 hits.")

    # --- 3. RRF Reranking ---
    results_map = {
        "vector": hits_vector,
        "bm25": hits_bm25
    }

    final_results = reciprocal_rank_fusion(results_map)
    return final_results[:top_k]

if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == 'exit':
                break

            results = search_sc_judgments(user_query)

            save_to_csv(user_query, results)

            print("\n" + "="*60)
            print(f"Top {len(results)} Results for: '{user_query}'")
            print("="*60)

            for i, hit in enumerate(results):
                fields = hit.get('fields', {})
                print(f"\n{i+1}. CASE: {fields.get('case_name', 'Unknown')}")
                print(f"   ID: {hit.get('id')}")
            print("\n")

    except KeyboardInterrupt:
        print("\nExiting...")


# import sys
# import json
# import csv
# import os
# from vespa.application import Vespa
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict
# import torch

# # --- Configuration ---
# VESPA_URL = "http://localhost:8085"
# MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
# RRF_K = 60
# CSV_FILENAME = "sc_search_results.csv"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --- Global Initialization ---
# # Load model ONCE here instead of inside the loop (much faster)
# print(f"Loading model {MODEL_NAME} on {DEVICE}...")
# model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
# model.max_seq_length = 8192

# def get_vespa_app():
#     return Vespa(url=VESPA_URL)

# def reciprocal_rank_fusion(results_dict: Dict[str, List[Dict]], k=RRF_K):
#     rrf_score = {}

#     for method, hits in results_dict.items():
#         for rank, hit in enumerate(hits):
#             doc_id = hit.get('id')
#             if doc_id not in rrf_score:
#                 rrf_score[doc_id] = {"score": 0, "hit": hit}
#             rrf_score[doc_id]["score"] += 1.0 / (k + rank + 1)

#     sorted_docs = sorted(rrf_score.values(), key=lambda x: x['score'], reverse=True)
#     return [item['hit'] for item in sorted_docs]

# def search_sc_judgments(query_text, top_k=5):
#     app = get_vespa_app()
#     print(f"Encoding query: '{query_text}'")

#     # Use the globally loaded model
#     query_emb = model.encode(query_text).tolist()

#     # --- 1. Vector Search ---
#     vector_res = app.query(body={
#         "yql": "select * from scjudgments where {targetHits:10}nearestNeighbor(embedding, q_emb)",
#         "ranking.profile": "semantic",
#         "input.query(q_emb)": query_emb,
#         "hits": 10
#     })

#     # --- 2. BM25 Search ---
#     bm25_res = app.query(body={
#         "yql": "select * from scjudgments where userQuery()",
#         "query": query_text,
#         "ranking.profile": "bm25",
#         "hits": 10
#     })

#     hits_vector = vector_res.hits if vector_res.hits else []
#     hits_bm25 = bm25_res.hits if bm25_res.hits else []

#     print(f"   -> Found {len(hits_vector)} vector hits and {len(hits_bm25)} BM25 hits.")

#     # --- 3. RRF Reranking ---
#     results_map = {
#         "vector": hits_vector,
#         "bm25": hits_bm25
#     }

#     final_results = reciprocal_rank_fusion(results_map)
#     return final_results[:top_k]

# def save_to_csv(query, results):
#     """Helper function to save results to CSV."""
#     file_exists = os.path.isfile(CSV_FILENAME)
    
#     csv_rows = []
#     for i, hit in enumerate(results):
#         fields = hit.get('fields', {})
#         csv_rows.append({
#             "Query": query,
#             "Rank": i + 1,
#             "ID": hit.get('id'),
#             "Case Name": fields.get('case_name', 'Unknown'),
#             "Date": fields.get('date', 'N/A'),
#             # Saving a snippet of text (first 500 chars) to keep CSV manageable
#             "Text Snippet": fields.get('text', '')[:500] 
#         })

#     if not csv_rows:
#         return

#     try:
#         with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=["Query", "Rank", "ID", "Case Name", "Date", "Text Snippet"])
            
#             # Write header only if file didn't exist
#             if not file_exists:
#                 writer.writeheader()
            
#             writer.writerows(csv_rows)
#             print(f" >> Saved {len(csv_rows)} rows to {CSV_FILENAME}")
#     except Exception as e:
#         print(f"Error saving CSV: {e}")

# if __name__ == "__main__":
#     try:
#         while True:
#             user_query = input("Enter search query (or 'exit'): ")
#             if user_query.lower() == 'exit':
#                 break

#             results = search_sc_judgments(user_query)

#             print("\n" + "="*60)
#             print(f"Top {len(results)} Results for: '{user_query}'")
#             print("="*60)

#             for i, hit in enumerate(results):
#                 fields = hit.get('fields', {})
#                 print(f"\n{i+1}. CASE: {fields.get('case_name', 'Unknown')}")
#                 print(f"   ID: {hit.get('id')}")
#             print("\n")

#             # Save results to CSV
#             save_to_csv(user_query, results)

#     except KeyboardInterrupt:
#         print("\nExiting...")