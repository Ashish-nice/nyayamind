import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections

# Load the embedding model
model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', device='cuda', trust_remote_code=True)

# Encode the query into a vector
query_text = "Can an advocate be a director in a company?"
query_vector = model.encode(query_text)

# Connect to Milvus
print("Connecting to Milvus...")
connections.connect(host="localhost", port="19530", alias="default")
print("Successfully connected.")

# Load the collection
collection_name = "Central_acts_db_latest"  # Corrected collection name
collection = Collection(name=collection_name)

# Load collection into memory before searching
print(f"Loading collection '{collection_name}'...")
collection.load()
print("Collection loaded.")

# Perform a similarity search
print("Performing search...")
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 32} # Adjust nprobe for a balance of speed and accuracy
}

results = collection.search(
    data=[query_vector],
    anns_field="Embedding",             # Corrected vector field name (case-sensitive)
    param=search_params,
    limit=10,                            # Retrieve top 5 results
    output_fields=["id",
                "Section_Text",
                "State_Name",
                "Name_of_Statute",
                "Section_Number",
                "Section_Title"]  # Corrected output fields
)
print("Search complete.")

# Display the results
print(f"\nTop 10 results for query: '{query_text}'")
print("="*40)
print("*"*40)
print(results[0])
print("*"*40)
results_list = []
for i, hit in enumerate(results[0]):
    print(f"--- Result {i+1} ---")
    print(f"Distance: {hit.distance:.4f}")
    print(f"State Name: {hit.entity.get('State_Name')}")
    print(f"Section_Number: {hit.entity.get('Section_Number')}")
    print(f"Section_Title: {hit.entity.get('Section_Title')}")
    print(f"Name_of_Statute: {hit.entity.get('Name_of_Statute')}")
    print(f"Text: {hit.entity.get('Section_Text')[:300]}...")
    print("-"*(len("--- Result {i+1} ---")-1))
    
    # Append result to list
    results_list.append({
        'query': query_text,
        'distance': f"{hit.distance:.4f}",
        'State_Name': hit.entity.get('State_Name'),
        'Section_Number': hit.entity.get('Section_Number'),
        'text_chunk': hit.entity.get('Section_Text'),
        'Name_of_Statute': hit.entity.get('Name_of_Statute'),
        'Section_Title': hit.entity.get('Section_Title')
    })

# Save results to a CSV file
results_df = pd.DataFrame(results_list)
output_filename = "query_results.csv"
results_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ Results successfully saved to '{output_filename}'")

# Disconnect from Milvus
connections.disconnect("default")
print("\nDisconnected from Milvus.")
