import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings)
from langchain_community.vectorstores import milvus
import _csv
from tqdm import tqdm
from langchain_community.vectorstores import Milvus
import pymilvus
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType, IndexType,
    Collection, AnnSearchRequest, RRFRanker, WeightedRanker, db
)
import time
import os
import unicodedata
import ast
import json
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

connection = connections.connect(host="localhost", port=19530)
mc = MilvusClient(connections=connection)


# Load model and tokenizer
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", device="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-v2.0")
embedding_dim = model.get_sentence_embedding_dimension()

def preprocess_vector(text):
    if pd.isna(text):
        return ""
    text = text.replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = ' '.join(text.split())
    return text

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, description='ids', max_length=100, is_primary=True, auto_id=False),
    FieldSchema(name='Embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
    FieldSchema(name="Text", dtype=DataType.VARCHAR, max_length=60000),
    FieldSchema(name="Case_Name", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="Section_Number", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="Section_Title", dtype=DataType.VARCHAR, max_length=20000),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="Embedding", index_params=index_params)
    print(f"Successfully created collection: `{collection_name}`")
    return collection

def isExceedsTokenLength(text, tokenizer, max_length=4096):
    tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)
    return len(tokens['input_ids'])


collection = create_milvus_collection('Central_Acts', embedding_dim)

# Text splitter with token-aware chunk length function
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,  # Large enough to allow initial splits, refined later
    chunk_overlap=100,  # Overlap not specified in requirements
    separators=[
        "\n\n",           # First level: double newline
        "\n \n",          # First level: newline, space, newline
        r"\n\d+\. ",". "      # Second level: \n followed by numbers, period, space
    ],
    length_function=lambda text: isExceedsTokenLength(text, tokenizer)
)
df = pd.read_csv('/home/shubham/tanmay/lexensemble/data/Central_Acts.csv')
data = []
db_id = 0
json_records = []

def conversion(document):
    document["id"] = int(document["id"])
    document["Text"] = str(document["Text"])[:60000]
    document["Case_Name"] = str(document["Case_Name"])[:20000]
    document["Section_Number"] = str(document["Section_Number"])[:1000]
    document["Section_Title"] = str(document["Section_Title"])[:20000]
    return document

for i in tqdm(range(len(df)), desc="Encoding Central Acts"):
    section_text = df['Section Text'].iloc[i]
    chunks = text_splitter.split_text(section_text)

    if (i%100) == 1:
        print(f"🔹 Original text token count: {isExceedsTokenLength(section_text, tokenizer)}")
        print(f"🔸 Number of chunks: {len(chunks)}")

    isMultiChunk = False
    if(len(chunks) > 1 or isExceedsTokenLength(section_text, tokenizer) > 4096):
        isMultiChunk = True

    for chunk in chunks:
        token_len = isExceedsTokenLength(chunk, tokenizer)
        if (i%100) == 1:
            print(f"  ⤷ Chunk token length: {token_len}")
        
        chunk = preprocess_vector(chunk)
        embedding = model.encode(chunk).astype(np.float32)

        if(isMultiChunk):
            if (i%100) == 1:
                print(f"  ⤷ Chunk: {chunk}")

        document = {
            "id": db_id,
            "Embedding": embedding,
            "Text": chunk,
            "Case_Name": str(df['Name of statute'].iloc[i]),
            "Section_Number": str(df['Section Number'].iloc[i]),
            "Section_Title": str(df['Section Title'].iloc[i]),
        }
        document = conversion(document)
        data.append(document)
        db_id += 1



        if len(data) >= 2000:
            collection.insert(data)
            data = []

if data:
    collection.insert(data)

collection.flush()

# Save JSON snapshot first (before loading collection)
schema_info = []
for field in collection.schema.fields:
    schema_info.append({
        "name": field.name,
        "dtype": str(field.dtype),
        "is_primary": field.is_primary,
        "auto_id": field.auto_id,
        "max_length": getattr(field, "max_length", None),
        "dim": getattr(field, "dim", None)
    })

json_output = {
    "collection_name": "Central_Acts_DB_latest",
    "schema": schema_info,
    "index_params": {
        'metric_type': 'L2',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 2048}
    },
    "data": json_records
}

json_path = "Central_Acts_DB_latest.json"
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(json_output, jf, ensure_ascii=False, indent=2)
print(f"📄 JSON snapshot (schema + data) written to {json_path}")

# Load collection into memory before querying
print(f"🔄 Loading collection Central_Acts_DB_latest into memory...")
collection.load()

# Wait a moment to ensure collection is fully loaded
time.sleep(2)

# Verify collection is loaded
if not collection.is_empty:
    print(f"✅ Collection Central_Acts_DB_latest loaded successfully with {collection.num_entities} entities")
else:
    print(f"⚠️ Collection Central_Acts_DB_latest appears to be empty")

# Fetch all records from Milvus (except embeddings)
output_fields = [
    "id",
    "Text",
    "Case_Name",
    "Section_Number",
    "Section_Title"
]

try:
    all_records = collection.query(expr="id >= 0", output_fields=output_fields)
    
    # Print a summary
    print(f"Total records in Milvus: {len(all_records)}")
    for rec in all_records[:5]:  # Print first 5 for brevity
        print(rec)

    # Save to JSON
    with open("milvus_actual_data_central_acts.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
        
except Exception as query_error:
    print(f"⚠️ Warning: Could not query collection: {query_error}")
    print(f"📊 Using JSON snapshot data instead (contains {len(json_records)} records)")
    
    # Use the JSON records we already have as fallback
    with open("milvus_actual_data_central_acts.json", "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)

print(f"✅ Completed indexing for Central Acts")
