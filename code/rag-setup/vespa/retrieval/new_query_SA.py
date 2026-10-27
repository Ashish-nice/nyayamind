import sys
from vespa.application import Vespa
from sentence_transformers import SentenceTransformer
import torch
import csv
import os

# --- Configuration ---
# 1. Define the list of states you want to query across.
# This must match the state names used during setup/ingestion.
STATE_LIST = ['sa_sample_part1', 'sa_sample_part2']
# Example full list:
# STATE_LIST = ['Lakshadweep', 'Madhya Pradesh', 'Tamil Nadu', ... ]

def get_all_state_sources():
    """Generates the comma-separated string of schema sources."""
    sources = []
    for state in STATE_LIST:
        clean_name = state.lower().replace(" ", "_")
        sources.append(f"sa_{clean_name}")
    return ", ".join(sources)

# 2. Connect to Vespa
app = Vespa(url="http://localhost:8085")

# 3. Load embedding model
model_name = "Snowflake/snowflake-arctic-embed-m-v2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def save_to_csv(query, results, filename="sa_search_results_multi.csv"):
    """Helper function to save results to CSV."""
    file_exists = os.path.isfile(filename)
    
    csv_rows = []
    for i, hit in enumerate(results):
        fields = hit.get('fields', {})
        
        doc_id = hit.get('id', '')
        source_display = doc_id.split(':')[2] if len(doc_id.split(':')) > 2 else "Unknown"

        # Prepare clean snippet
        content = fields.get('section_text', '')
        clean_content = " ".join(content.split())[:1000] # Limit to 1000 chars for CSV
        
        csv_rows.append({
            "Query": query,
            "Rank": i + 1,
            "State": fields.get('state_name', 'Unknown State'),
            "Statute": fields.get('name_of_statute', 'Unknown Statute'),
            "Section Number": fields.get('section_number', ''),
            "Section Title": fields.get('section_title', ''),
            "Section Text Snippet": clean_content,
            "source": source_display
        })

    if not csv_rows:
        return

    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Query", "Rank", "State", "Statute", "Section Number", "Section Title", "Section Text Snippet", "source"])
            
            # Write header only if file didn't exist previously
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(csv_rows)
            print(f" >> Saved {len(csv_rows)} rows to {filename}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, hit in enumerate(vector_results):
        doc_id = hit['id']
        if doc_id not in scores:
            scores[doc_id] = {'hit': hit, 'score': 0}
        scores[doc_id]['score'] += 1 / (k + rank + 1)

    for rank, hit in enumerate(bm25_results):
        doc_id = hit['id']
        if doc_id not in scores:
            scores[doc_id] = {'hit': hit, 'score': 0}
        scores[doc_id]['score'] += 1 / (k + rank + 1)

    sorted_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    return [item['hit'] for item in sorted_docs]

def search_state_acts(query_text):
    print(f"Querying State Acts for: '{query_text}'")

    # Generate the source string for YQL (e.g., "sa_kerala, sa_delhi")
    target_sources = get_all_state_sources()
    if not target_sources:
        print("No state schemas defined in configuration.")
        return

    query_vector = model.encode(query_text).tolist()

    # --- 1. Vector Search ---
    try:
        # We select from 'sources' followed by the list of our generated schema names
        yql_vector = f"select * from sources {target_sources} where ({{targetHits:10}}nearestNeighbor(embedding, q_emb));"

        vector_response = app.query(
            body={
                "yql": yql_vector,
                "ranking": "semantic",
                "input.query(q_emb)": query_vector,
                "hits": 10
            }
        )
        vector_hits = vector_response.hits
    except Exception as e:
        print(f"Error during vector search: {e}")
        vector_hits = []

    print(f"   -> Vector matches: {len(vector_hits)}")

    # --- 2. BM25 Search ---
    try:
        yql_bm25 = f"select * from sources {target_sources} where userQuery();"

        bm25_response = app.query(
            body={
                "yql": yql_bm25,
                "query": query_text,
                "ranking": "bm25",
                "hits": 10
            }
        )
        bm25_hits = bm25_response.hits
    except Exception as e:
        print(f"Error during BM25 search: {e}")
        bm25_hits = []

    print(f"   -> BM25 matches: {len(bm25_hits)}")

    # --- 3. Rerank ---
    reranked_results = reciprocal_rank_fusion(vector_hits, bm25_hits)

    # --- 4. Display ---
    print("\nTop 5 State Act Results:")
    print("-" * 50)

    save_to_csv(query_text, reranked_results[:5])

    for i, hit in enumerate(reranked_results[:5]):
        fields = hit.get('fields', {})

        doc_id = hit.get('id', '')
        source_display = doc_id.split(':')[2] if len(doc_id.split(':')) > 2 else "Unknown"

        state = fields.get('state_name', 'Unknown State')
        statute = fields.get('name_of_statute', 'Unknown Statute')
        sec_num = fields.get('section_number', '')
        sec_title = fields.get('section_title', '')
        content = fields.get('section_text', '')

        print(f"{i+1}. [{state}] {statute}")
        print(f"   Section {sec_num}: {sec_title}")
        print(f"   ID: {doc_id} | Source: {source_display}")

        if content:
            clean_content = " ".join(content.split())
            print(f"   Snippet: {clean_content[:150]}...")
        print("-" * 50)

if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter search query (or 'exit'): ")
            if user_query.lower() == 'exit':
                break
            search_state_acts(user_query)
    except KeyboardInterrupt:
        print("\nExiting...")
