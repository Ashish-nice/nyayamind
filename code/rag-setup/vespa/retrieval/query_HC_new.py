import csv
import os
from vespa.application import Vespa
from sentence_transformers import SentenceTransformer
import torch

app = Vespa(url="http://localhost:8085")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", device=device, trust_remote_code=True)
model.max_seq_length = 8192

# List of all High Court schemas/collections created in ingestion
HC_SOURCES = ['HC_Judgments_DB', #,
    #'HC_Allahabad', 'HC_Kolkata', 'HC_Guwahati', 'HC_Telangana',
    #'HC_AndhraPradesh', 'HC_Chhattisgarh', 'HC_Gujarat', 'HC_HimachalPradesh',
    #'HC_JammuAndKashmir', 'HC_Jharkhand', 'HC_Karnataka', 'HC_Kerala',
    #'HC_MadhyaPradesh', 'HC_Manipur', 'HC_Meghalaya', 'HC_Orissa',
    #'HC_PunjabAndHaryana', 'HC_Rajasthan', 'HC_Sikkim', 'HC_Tripura',
    #'HC_Uttarakhand', 'HC_Madras', 'HC_Patna', 'HC_Delhi'
]

SOURCES_STR = ", ".join(HC_SOURCES)
print(f"Sources configured: {SOURCES_STR}")

def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    doc_data = {}

    for results in results_list:
        for rank, hit in enumerate(results):
            doc_id = hit['id']
            if doc_id not in doc_data:
                doc_data[doc_id] = hit

            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs, doc_data

def search_hybrid_rrf(query_text):
    print(f"Querying across {len(HC_SOURCES)} High Courts: {query_text}")

    # 1. Semantic Search
    query_emb = model.encode(query_text).tolist()
    vector_yql = f"select * from sources {SOURCES_STR} where {{targetHits:10}}nearestNeighbor(embedding, q_emb)"

    vector_res = app.query(
        body={
            "yql": vector_yql,
            "ranking.profile": "semantic",
            "input.query(q_emb)": query_emb,
            "hits": 10
        }
    )
    vector_hits = vector_res.hits if vector_res.hits else []

    # 2. BM25 Search
    bm25_yql = f"select * from sources {SOURCES_STR} where userQuery()"

    bm25_res = app.query(
        body={
            "yql": bm25_yql,
            "query": query_text,
            "ranking.profile": "bm25",
            "hits": 10
        }
    )
    bm25_hits = bm25_res.hits if bm25_res.hits else []

    # 3. RRF Reranking
    ranked_tuples, doc_map = reciprocal_rank_fusion([vector_hits, bm25_hits])

    # 4. Display & Collect Top 5
    print(f"\n--- Top 5 Results (RRF) ---")
    
    csv_rows = [] # List to store data for CSV
    
    for i, (doc_id, score) in enumerate(ranked_tuples[:5]):
        doc = doc_map[doc_id]
        fields = doc.get('fields', {})
        source_hc = doc_id.split(':')[1] if ':' in doc_id else "Unknown"

        # Print to console
        print(f"Rank {i+1} | Score: {score:.4f} | Source: {source_hc}")
        print(f"Case: {fields.get('case_name', 'N/A')}")
        print(f"Bench: {fields.get('bench', 'N/A')}")
        snippet = fields.get('text', '')[:100]
        print(f"Snippet: {snippet}...")
        print("-" * 50)
        
        # Add to CSV list
        csv_rows.append({
            "Query": query_text,
            "Rank": i + 1,
            "Score": round(score, 4),
            "Source": source_hc,
            "Case Name": fields.get('case_name', 'N/A'),
            "Bench": fields.get('bench', 'N/A'),
            "Full Text": fields.get('text', '') # Saving full text to CSV
        })

    # 5. Save to CSV
    filename = "hc_search_results_multi.csv"
    file_exists = os.path.isfile(filename)

    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Query", "Rank", "Score", "Source", "Case Name", "Bench", "Full Text"])
            
            # Write header only if file didn't exist previously
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(csv_rows)
            print(f" >> Results appended to {filename}")
            
    except Exception as e:
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == 'exit':
                break
            search_hybrid_rrf(user_query)
    except KeyboardInterrupt:
        print("\nExiting...")