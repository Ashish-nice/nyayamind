import pandas as pd
from sentence_transformers import SentenceTransformer
from endee import Endee  

print("📦 Loading embedding model...")
model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', device='cuda', trust_remote_code=True)

query_text = "Court fees for commercial cases above 1 crore."
query_vector = model.encode(query_text).tolist()  # Ensure it's a list

print("🔌 Connecting to Endee...")
client = Endee(token="shubham:NXdEaTCJQEmqnPTKicXNVDFRRvuryuAR")
client.set_base_url("http://172.27.21.35:8080/api/v1")  # <-- Replace with your actual API token
print("✅ Connected to Endee.")

collection = client.get_index("State_Acts")

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
    print(f"State Name: {meta.get('State_Name')}")
    print(f"Section_Number: {meta.get('Section_Number')}")
    print(f"Section_Title: {meta.get('Section_Title')}")
    print(f"Name_of_Statute: {meta.get('Name_of_Statute')}")
    print(f"Text: {meta.get('Section_Text', '')[:300]}...")
    print("-" * 40)
    results_list.append({
        'query': query_text,
        'similarity': f"{hit.get('similarity', 0):.4f}",
        'State_Name': meta.get('State_Name'),
        'Section_Number': meta.get('Section_Number'),
        'text_chunk': meta.get('Section_Text'),
        'Name_of_Statute': meta.get('Name_of_Statute'),
        'Section_Title': meta.get('Section_Title')
    })

results_df = pd.DataFrame(results_list)
output_filename = "sa_query_2_results_vecx.csv"
results_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ Results successfully saved to '{output_filename}'")