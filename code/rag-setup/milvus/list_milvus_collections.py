from pymilvus import utility, connections

def list_milvus_collections():
    """
    Connects to a Milvus instance and lists all available collections.
    """
    try:
        # 1. Connect to Milvus
        # Ensure your Milvus instance is running before executing this.
        print("Connecting to Milvus...")
        connections.connect(host="localhost", port="19530", alias="default")
        print("Successfully connected to Milvus.")

        # 2. List all collections
        collections = utility.list_collections()
        
        if collections:
            print("\n📁 Found the following collections:")
            for i, collection_name in enumerate(collections, 1):
                print(f"   {i}. {collection_name}")
        else:
            print("\nNo collections found in the database.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 3. Disconnect from Milvus
        connections.disconnect('default')
        print("\nDisconnected from Milvus.")

if __name__ == "__main__":
    list_milvus_collections()