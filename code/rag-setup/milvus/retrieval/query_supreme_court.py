import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections

# Step 1: Load the embedding model
print("📦 Loading embedding model...")
model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', device='cuda', trust_remote_code=True)

# Step 2: Encode the query
query_text = "What is the court's standing on discrimination related to sex workers and their children?"
query_vector = model.encode(query_text)

# Step 3: Connect to Milvus
print("🔌 Connecting to Milvus...")
connections.connect(host="localhost", port="19530", alias="default")
print("✅ Connected to Milvus.")

# Step 4: Load the target collection
collection_name = "SC_Judgments_DB_latest_V2"
collection = Collection(name=collection_name)

# Step 5: Load collection into memory
print(f"📂 Loading collection: {collection_name}...")
collection.load()
print("✅ Collection loaded.")

# Step 6: Perform semantic similarity search
print("🔍 Performing semantic search...")
search_params = {
    "metric_type": "L2",
    "params": {
        "ef": 64  # Try 64, 100, or higher for better recall (at the cost of speed)
    }
}

results = collection.search(
    data=[query_vector],
    anns_field="Embedding",
    param=search_params,
    limit=10,
    output_fields=[
        "id", "Text", "Case_Name", "Diary_Number", "Judgement_Type", "Case_Number",
        "Petitioner", "Respondent", "Petitioner_Advocate", "Respondent_Advocate",
        "Bench", "Judgment_date", "Citations"
    ]
)

# Step 7: Display and collect results
print(f"\n🔝 Top 10 results for query: '{query_text}'")
print("=" * 60)
results_list = []
for i, hit in enumerate(results[0]):
    print(f"--- Result {i+1} ---")
    print(f"Distance: {hit.distance:.4f}")
    print(f"Case Name: {hit.entity.get('Case_Name')}")
    print(f"Date: {hit.entity.get('Judgment_date')}")
    print(f"Bench: {hit.entity.get('Bench')}")
    print(f"Petitioner: {hit.entity.get('Petitioner')}")
    print(f"Respondent: {hit.entity.get('Respondent')}")
    print(f"Judgment Text (first 300 chars):\n{hit.entity.get('Text')[:300]}...")
    print("-" * 60)
    
    results_list.append({
        "query": query_text,
        "distance": f"{hit.distance:.4f}",
        "Case_Name": hit.entity.get("Case_Name"),
        "Judgment_date": hit.entity.get("Judgment_date"),
        "Bench": hit.entity.get("Bench"),
        "Petitioner": hit.entity.get("Petitioner"),
        "Respondent": hit.entity.get("Respondent"),
        "Judgement_Type": hit.entity.get("Judgement_Type"),
        "Text_Chunk": hit.entity.get("Text"),
        "Citations": hit.entity.get("Citations")
    })

# Step 8: Save results to CSV
output_filename = "sc_judgment_query_results_q3.csv"
pd.DataFrame(results_list).to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ Results saved to '{output_filename}'")

# Step 9: Disconnect
connections.disconnect("default")
print("🔌 Disconnected from Milvus.")