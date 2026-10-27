import pandas as pd
from sentence_transformers import SentenceTransformer
from endee import Endee  

print("📦 Loading embedding model...")
model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', device='cuda', trust_remote_code=True)

query_text = "charge alleging that he had deserted his wife and two school going children and was residing along with another lady."
query_vector = model.encode(query_text).tolist()  # Ensure it's a list

print("🔌 Connecting to Endee...")
client = Endee(token="shubham:NXdEaTCJQEmqnPTKicXNVDFRRvuryuAR")
client.set_base_url("http://172.27.21.35:8080/api/v1")  # <-- Replace with your actual API token
print("✅ Connected to Endee.")

collection = client.get_index("SC_Judgments_DB")

print("🔍 Performing semantic search...")
results = collection.query(
    vector=query_vector,
    top_k=10,
    ef=128,
    include_vectors=False  
)

print(f"\n🔝 Top 10 results for query: '{query_text}'")
print("=" * 60)
results_list = []
for i, hit in enumerate(results):
    meta = hit.get("meta", {})
    print(f"--- Result {i+1} ---")
    print(f"Similarity: {hit.get('similarity', 0):.4f}")
    print(f"Case Name: {meta.get('Case_Name')}")
    print(f"Date: {meta.get('Judgment_date')}")
    print(f"Bench: {meta.get('Bench')}")
    print(f"Petitioner: {meta.get('Petitioner')}")
    print(f"Respondent: {meta.get('Respondent')}")
    print(f"Judgment Text (first 300 chars):\n{meta.get('Text', '')[:300]}...")
    print("-" * 60)
    results_list.append({
        "query": query_text,
        "similarity": f"{hit.get('similarity', 0):.4f}",
        "Case_Name": meta.get("Case_Name"),
        "Judgment_date": meta.get("Judgment_date"),
        "Bench": meta.get("Bench"),
        "Petitioner": meta.get("Petitioner"),
        "Respondent": meta.get("Respondent"),
        "Judgement_Type": meta.get("Judgement_Type"),
        "Text_Chunk": meta.get("Text"),
        "Citations": meta.get("Citations")
    })

output_filename = "sc_query_2_results_vecx.csv"
pd.DataFrame(results_list).to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ Results saved to '{output_filename}'")