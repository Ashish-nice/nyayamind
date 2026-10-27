# import sys
# from sentence_transformers import SentenceTransformer
# from vespa.application import Vespa
# import torch

# # --- Configuration ---
# VESPA_URL = "http://localhost:8085"
# # Using the same model as defined in your setup/previous context
# MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
# RRF_K = 60

# # Initialize
# app = Vespa(url=VESPA_URL)
# print("Loading model for query encoding...")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)

# def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
#     """
#     Computes RRF score to combine Vector and BM25 lists.
#     """
#     scores = {}

#     # Process Vector Results
#     for rank, hit in enumerate(vector_results):
#         doc_id = hit['id']
#         scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))

#     # Process BM25 Results
#     for rank, hit in enumerate(bm25_results):
#         doc_id = hit['id']
#         scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))

#     # Sort by score descending
#     return sorted(scores.items(), key=lambda item: item[1], reverse=True)

# def search_central_acts(query_text):
#     print(f"\nSearching Central Acts for: '{query_text}'")

#     query_emb = model.encode(query_text).tolist()

#     # 1. Vector Search (Central Acts Only)
#     # Uses ranking profile 'vector_only' defined in your ca_schema
#     vector_response = app.query(body={
#         "yql": "select * from sources centralacts where ({targetHits:10}nearestNeighbor(embedding,q_emb));",
#         "ranking": "semantic",
#         # FIX: "input.query(q_emb)" must be at the top level, not nested in "inputs"
#         "input.query(q_emb)": query_emb,
#         "hits": 10
#     })
#     vector_hits = vector_response.hits
#     print(f"   Found {len(vector_hits)} vector matches")

#     # 2. BM25 Search (Central Acts Only)
#     # Uses ranking profile 'bm25_only' defined in your ca_schema
#     bm25_response = app.query(body={
#         "yql": "select * from sources centralacts where userQuery();",
#         "query": query_text,
#         "type": "all",
#         "ranking": "bm25",
#         "hits": 10
#     })
#     bm25_hits = bm25_response.hits
#     print(f"   Found {len(bm25_hits)} BM25 matches")

#     # 3. RRF Re-ranking
#     reranked_ids = reciprocal_rank_fusion(vector_hits, bm25_hits, k=RRF_K)
#     top_5_ids = [doc_id for doc_id, score in reranked_ids[:5]]

#     print("\n--- Top 5 Central Acts Results ---")

#     # Create a lookup map for printing
#     all_hits_map = {h['id']: h for h in vector_hits + bm25_hits}

#     for rank, doc_id in enumerate(top_5_ids):
#         doc = all_hits_map.get(doc_id)
#         if doc:
#             fields = doc['fields']

#             # Extract fields specific to Central Acts schema
#             act_name = fields.get('name_of_statute', 'Unknown Act')
#             sec_num = fields.get('section_number', '?')
#             sec_title = fields.get('section_title', 'Untitled')
#             text = fields.get('section_text', '')

#             print(f"{rank+1}. [{doc_id}] {act_name} - Section {sec_num}")
#             print(f"   Title: {sec_title}")
#             print(f"   Snippet: {text[:200]}...\n")

# if __name__ == "__main__":
#     try:
#         while True:
#             user_query = input("Enter search query (or 'exit'): ")
#             if user_query.lower() == 'exit':
#                 break
#             search_central_acts(user_query)
#     except KeyboardInterrupt:
#         print("\nExiting...")


import sys
import csv
import os
from sentence_transformers import SentenceTransformer
from vespa.application import Vespa
import torch

# --- Configuration ---
VESPA_URL = "http://localhost:8085"
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
RRF_K = 60
CSV_FILENAME = "ca_search_results.csv"

# Initialize
app = Vespa(url=VESPA_URL)
print("Loading model for query encoding...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """
    Computes RRF score to combine Vector and BM25 lists.
    """
    scores = {}

    # Process Vector Results
    for rank, hit in enumerate(vector_results):
        doc_id = hit['id']
        scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))

    # Process BM25 Results
    for rank, hit in enumerate(bm25_results):
        doc_id = hit['id']
        scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))

    # Sort by score descending
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def save_to_csv(query, results_list):
    """
    Saves the list of result dictionaries to a CSV file.
    """
    file_exists = os.path.isfile(CSV_FILENAME)
    
    try:
        with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = ["Query", "Rank", "Act Name", "Section Number", "Section Title", "Snippet"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerows(results_list)
            print(f" >> Saved {len(results_list)} rows to {CSV_FILENAME}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def search_central_acts(query_text):
    print(f"\nSearching Central Acts for: '{query_text}'")

    query_emb = model.encode(query_text).tolist()

    # 1. Vector Search (Central Acts Only)
    vector_response = app.query(body={
        "yql": "select * from sources centralacts where ({targetHits:10}nearestNeighbor(embedding,q_emb));",
        "ranking": "semantic",
        "input.query(q_emb)": query_emb,
        "hits": 10
    })
    vector_hits = vector_response.hits
    print(f"   Found {len(vector_hits)} vector matches")

    # 2. BM25 Search (Central Acts Only)
    bm25_response = app.query(body={
        "yql": "select * from sources centralacts where userQuery();",
        "query": query_text,
        "type": "all",
        "ranking": "bm25",
        "hits": 10
    })
    bm25_hits = bm25_response.hits
    print(f"   Found {len(bm25_hits)} BM25 matches")

    # 3. RRF Re-ranking
    reranked_ids = reciprocal_rank_fusion(vector_hits, bm25_hits, k=RRF_K)
    top_5_ids = [doc_id for doc_id, score in reranked_ids[:5]]

    print("\n--- Top 5 Central Acts Results ---")

    # Create a lookup map
    all_hits_map = {h['id']: h for h in vector_hits + bm25_hits}
    csv_data = []

    for rank, doc_id in enumerate(top_5_ids):
        doc = all_hits_map.get(doc_id)
        if doc:
            fields = doc['fields']

            act_name = fields.get('name_of_statute', 'Unknown Act')
            sec_num = fields.get('section_number', '?')
            sec_title = fields.get('section_title', 'Untitled')
            text = fields.get('section_text', '')

            # Display to console
            print(f"{rank+1}. [{doc_id}] {act_name} - Section {sec_num}")
            print(f"   Title: {sec_title}")
            print(f"   Snippet: {text[:200]}...\n")

            # Prepare for CSV (clean text to avoid CSV breakage)
            clean_snippet = " ".join(text.split())[:1000]
            
            csv_data.append({
                "Query": query_text,
                "Rank": rank + 1,
                "Act Name": act_name,
                "Section Number": sec_num,
                "Section Title": sec_title,
                "Snippet": clean_snippet
            })

    # 4. Save to CSV
    if csv_data:
        save_to_csv(query_text, csv_data)

if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == 'exit':
                break
            search_central_acts(user_query)
    except KeyboardInterrupt:
        print("\nExiting...")