import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Milvus
import os
import unicodedata
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, MilvusClient
)
import json

# Central place to keep index settings so we can also export them
INDEX_PARAMS = {
    'metric_type': 'L2',
    'index_type': 'IVF_FLAT',
    'params': {'nlist': 2048}
}

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

def isExceedsTokenLength(text, tokenizer, max_length=4096):
    tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)
    return len(tokens['input_ids'])

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, description='ids', max_length=100, is_primary=True, auto_id=False),
        FieldSchema(name='Embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
        FieldSchema(name="original_doc_id", dtype=DataType.INT64, description='original document id from csv'),
        FieldSchema(name="State_Name", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="Name_of_Statute", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="Section_Number", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Section_Title", dtype=DataType.VARCHAR, max_length=8000),
        FieldSchema(name="Section_Text", dtype=DataType.VARCHAR, max_length=40000),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)

    collection.create_index(
        field_name="Embedding",
        index_params={"index_type": "FLAT", "metric_type": "L2"}
    )
    print(f"✅ Successfully created collection: `{collection_name}`")
    return collection

def conversion(document):
    document["id"] = int(document["id"])
    document["original_doc_id"] = int(document["original_doc_id"])
    document["State_Name"] = str(document["State_Name"])[:20000]
    document["Name_of_Statute"] = str(document["Name_of_Statute"])[:20000]
    document["Section_Number"] = str(document["Section_Number"])[:4000]
    document["Section_Title"] = str(document["Section_Title"])[:8000]
    document["Section_Text"] = str(document["Section_Text"])[:40000]
    return document

# Milvus connection
connections.connect(host="localhost", port=19530)
mc = MilvusClient()

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


state_list = ['Lakshadweep', 'Madhya Pradesh', 'Tamil Nadu', 'Chandigarh', 'Bihar', 'Uttarakhand', 'Rajasthan', 'Jharkhand', 'Jammu and Kashmir', 'Odisha', 'Puducherry', 'Punjab', 'Andhra Pradesh', 'West Bengal', 'Tripura', 'Ladakh', 'Chhattisgarh', 'Maharashtra', 'Telangana', 'Himachal Pradesh', 'Gujarat', 'Manipur', 'Karnataka', 'Arunachal Pradesh', 'Andaman and Nicobar Islands', 'Assam', 'Delhi', 'Goa', 'Meghalaya', 'Dadra and Nagar Haveli and Daman and Diu', 'Kerala', 'Haryana', 'Uttar Pradesh']

for stateName in state_list:
    file_path = f'/home/shubham/tanmay/lexensemble/code/state_acts_csv/{stateName}.csv'
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping `{stateName}` — CSV not found at {file_path}")
        continue

    try:
        print(f"\n🔍 Processing state: {stateName}")
        collection_name = 'test_state_acts_' + stateName.replace(" ", "_")
        collection = create_milvus_collection(collection_name, embedding_dim)

        df = pd.read_csv(file_path)
        
        # Debug: Check column names and first few rows
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few rows of 'id' column: {df['id'].head().tolist()}")
        print(f"Data types: {df.dtypes}")
        
        data = []
        db_id = 0
        json_records = []

        for i in range(len(df)):
            section_text = df['Section Text'].iloc[i]
            chunks = text_splitter.split_text(section_text)

            if(i%40==1):
                print(f"🔹 Original text token count: {isExceedsTokenLength(section_text, tokenizer)}")
                print(f"🔸 Number of chunks: {len(chunks)}")

            isMultiChunk = False
            if(len(chunks) > 1 or isExceedsTokenLength(section_text, tokenizer) > 4096):
                # print(f"🔹 Original text: {section_text}")
                isMultiChunk = True


            for chunk in chunks:
                token_len = isExceedsTokenLength(chunk, tokenizer)
                if(i%40==1):
                    print(f"  ⤷ Chunk token length: {token_len}")
                
                chunk = preprocess_vector(chunk)
                embedding = model.encode(chunk).astype(np.float32)

                document = {
                    "id": db_id,
                    "original_doc_id": int(df["id"].iloc[i]),  # Store original document ID
                    "Embedding": embedding,
                    "State_Name": str(df["State Name"].iloc[i]),
                    "Name_of_Statute": str(df["Name of Statute"].iloc[i]),
                    "Section_Number": str(df["Section Number"].iloc[i]),
                    "Section_Title": str(df["Section Title"].iloc[i]),
                    "Section_Text": chunk,
                }
                document = conversion(document)
                data.append(document)
                db_id += 1

                json_doc = document.copy()
                if isinstance(json_doc.get("Embedding"), np.ndarray):
                    json_doc["Embedding"] = json_doc["Embedding"].tolist()
                if "Embedding" in json_doc:
                    del json_doc["Embedding"]
                json_records.append(json_doc)

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
            "collection_name": collection_name,
            "schema": schema_info,
            "index_params": INDEX_PARAMS,
            "data": json_records
        }

        json_path = os.path.join(os.path.dirname(file_path), f"{collection_name}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_output, jf, ensure_ascii=False, indent=2)
        print(f"📄 JSON snapshot (schema + data) written to {json_path}")

        # Load collection into memory before querying
        print(f"🔄 Loading collection {collection_name} into memory...")
        collection.load()
        
        # Wait a moment to ensure collection is fully loaded
        time.sleep(2)
        
        # Verify collection is loaded
        if not collection.is_empty:
            print(f"✅ Collection {collection_name} loaded successfully with {collection.num_entities} entities")
        else:
            print(f"⚠️ Collection {collection_name} appears to be empty")

        # Fetch all records from Milvus (except embeddings)
        output_fields = [
            "id",
            "original_doc_id",
            "State_Name",
            "Name_of_Statute",
            "Section_Number",
            "Section_Title",
            "Section_Text"
        ]

        try:
            all_records = collection.query(expr="id >= 0", output_fields=output_fields)
            
            # Option 1: Print to console
            print(f"Total records in Milvus: {len(all_records)}")
            for rec in all_records[:5]:  # Print first 5 for brevity
                print(rec)

            # Option 2: Save to JSON
            with open(f"milvus_actual_data_{stateName}.json", "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
                
        except Exception as query_error:
            print(f"⚠️ Warning: Could not query collection: {query_error}")
            print(f"📊 Using JSON snapshot data instead (contains {len(json_records)} records)")
            
            # Use the JSON records we already have as fallback
            with open(f"milvus_actual_data_{stateName}.json", "w", encoding="utf-8") as f:
                json.dump(json_records, f, ensure_ascii=False, indent=2)

        print(f"✅ Completed indexing for {stateName}")

    except Exception as e:
        print(f"❌ Error processing `{stateName}`: {str(e)}")
