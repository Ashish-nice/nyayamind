from pymilvus import connections, utility, Collection
import numpy as np
import logging 

logging.basicConfig(
    filename="milvus-info.log",
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

log = logging.getLogger()

log.info("Connecting to Milvus...")

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# List all collections
collections = utility.list_collections()
log.info(f"Found {len(collections)} collections:\n")

for name in collections:
    log.info(f"--- Processing Collection: {name} ---")
    try:
        collection = Collection(name)
        collection.load()
        
        # Schema
        log.info("Schema details:")
        dim = None
        for field in collection.schema.fields:
            log.info(f"Field:- {field.name} Type:-  ({field.dtype})")
            if field.dtype == 101:  # FLOAT_VECTOR
                dim = field.params.get("dim", None)

        # Number of entities
        count = collection.num_entities
        log.info(f"Number of entities: {count}")

        # Index details
        log.info("Indexes details:")
        for index in collection.indexes:
            log.info(f" - Index Field: {index.field_name}, Index Type: {index.params['index_type']}")

        # Estimate size (assuming float32 vectors = 4 bytes per dimension)
        if dim is not None:
            estimated_bytes = count * 768 * 4  # 4 bytes for float32
            estimated_mb = estimated_bytes / (1024 * 1024)
            log.info(f"Estimated size: {estimated_mb:.2f} MB")
    except Exception as e:
        log.error(f"Error loading collection {name}: {e}")
