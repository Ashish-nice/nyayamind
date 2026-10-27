import pandas as pd
from sentence_transformers import SentenceTransformer
from endee import Endee  

print("📦 Loading embedding model...")
model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', device='cuda', trust_remote_code=True)

query_text = "What deductions of pay are allowed for people employed by the Navy?"
query_vector = model.encode(query_text).tolist()  # Ensure it's a list

print("🔌 Connecting to Endee...")
client = Endee(token="shubham:NXdEaTCJQEmqnPTKicXNVDFRRvuryuAR")
client.set_base_url("http://172.27.21.35:8080/api/v1")
print("✅ Connected to Endee.")

collection = client.get_index("Central_Acts")

print("🔍 Performing semantic search...")
results = collection.query(
    vector=query_vector,
    top_k=10,
    ef=128,
    include_vectors=False
)

print(f"\n🔝 Top 10 results for query: '{query_text}'")
print("=" * 40)
results_list = []
for i, hit in enumerate(results):
    meta = hit.get("meta", {})
    print(f"--- Result {i+1} ---")
    print(f"Similarity: {hit.get('similarity', 0):.4f}")
    print(f"Case Name: {meta.get('Case_Name')}")
    print(f"Section_Number: {meta.get('Section_Number')}")
    print(f"Section_Title: {meta.get('Section_Title')}")
    print(f"Text: {meta.get('Text', '')[:300]}...")
    print("-" * 40)
    results_list.append({
        'query': query_text,
        'similarity': f"{hit.get('similarity', 0):.4f}",
        'Case_Name': meta.get('Case_Name'),
        'Section_Number': meta.get('Section_Number'),
        'Section_Title': meta.get('Section_Title'),
        'text_chunk': meta.get('Text')
    })

results_df = pd.DataFrame(results_list)
output_filename = "ca_query_3_results_vecx.csv"
results_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ Results successfully saved to '{output_filename}'")