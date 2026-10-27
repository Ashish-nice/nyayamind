from pymilvus import connections, utility

# Step 1: Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Step 2: Specify collection name
collection_name = "HC_test_db_exp"

# Step 3: Check if collection exists
if utility.has_collection(collection_name):
    # Step 4: Drop the collection
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' has been deleted.")
else:
    print(f"Collection '{collection_name}' does not exist.")
