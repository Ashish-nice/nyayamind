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
        FieldSchema(name="Case_Number", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="PDF_Path", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="State", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Disposal_Nature", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Decision_Date", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Text", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="Title", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Judge", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Date_of_Registration", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="CNR", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="Court_Name", dtype=DataType.VARCHAR, max_length=4000),
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
    document["Court_Name"] = str(document["Court_Name"])[:4000]
    document["Title"] = str(document["Title"])[:4000]
    document["Judge"] = str(document["Judge"])[:4000]
    document["CNR"] = str(document["CNR"])[:4000]
    document["Date_of_Registration"] = str(document["Date_of_Registration"])[:4000]
    document["Decision_Date"] = str(document["Decision_Date"])[:4000]
    document["Disposal_Nature"] = str(document["Disposal_Nature"])[:4000]
    document["State"] = str(document["State"])[:4000]
    document["PDF_Path"] = str(document["PDF_Path"])[:4000]
    document["Text"] = str(document["Text"])[:20000]
    document["Case_Number"] = str(document["Case_Number"])[:4000]
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

file_paths = [
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Calcutta_High_Court.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Tripura.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_for_State_of_Telangana.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Orissa.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Meghalaya.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Andhra_Pradesh.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Allahabad_High_Court.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Kerala.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Punjab_and_Haryana.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Rajasthan.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Gujarat.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Bombay_High_Court.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Himachal_Pradesh.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Uttarakhand.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Patna_High_Court.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Madhya_Pradesh.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Gauhati_High_Court.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Sikkim.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Jammu_and_Kashmir.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Madras_High_Court.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Delhi.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Karnataka.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Manipur.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Chhattisgarh.csv",
    "/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Jharkhand.csv"
]

collection_map = {'/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Allahabad_High_Court.csv': 'HC_Allahabad',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Calcutta_High_Court.csv': 'HC_Kolkata',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Gauhati_High_Court.csv': 'HC_Guwahati',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_for_State_of_Telangana.csv': 'HC_Telangana',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Andhra_Pradesh.csv': 'HC_AndhraPradesh',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Chhattisgarh.csv': 'HC_Chhattisgarh',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Gujarat.csv': 'HC_Gujarat',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Himachal_Pradesh.csv': 'HC_HimachalPradesh',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Jammu_and_Kashmir.csv': 'HC_JammuAndKashmir',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Jharkhand.csv': 'HC_Jharkhand',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Karnataka.csv': 'HC_Karnataka',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Kerala.csv': 'HC_Kerala',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Madhya_Pradesh.csv': 'HC_MadhyaPradesh',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Manipur.csv': 'HC_Manipur',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Meghalaya.csv': 'HC_Meghalaya',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Orissa.csv': 'HC_Orissa',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Punjab_and_Haryana.csv': 'HC_PunjabAndHaryana',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Rajasthan.csv': 'HC_Rajasthan',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Sikkim.csv': 'HC_Sikkim',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Tripura.csv': 'HC_Tripura',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_High_Court_of_Uttarakhand.csv': 'HC_Uttarakhand',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Madras_High_Court.csv': 'HC_Madras',
 '/home/shubham/tanmay/lexensemble/code/csv_files_by_hc/cases_new_summary_Patna_High_Court.csv': 'HC_Patna'}


for file_path in file_paths: 
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping `Bombay_High_Court` — CSV not found at {file_path}")
        continue

    try:
        collection_name = collection_map[file_path]
        print(f"\n🔍 Processing state: {collection_name}")
        collection = create_milvus_collection(collection_name, embedding_dim)

        df = pd.read_csv(file_path)
        data = []
        db_id = 0
        json_records = []

        for i in range(len(df)):
            section_text = df['Text'].iloc[i]

            if pd.isna(section_text) or not isinstance(section_text, str) or section_text.strip() == "":
                print(f"⚠️ Skipping row {i} due to invalid or empty text")
                continue

            chunks = text_splitter.split_text(section_text)

            isMultiChunk = False
            if(len(chunks) > 1 or isExceedsTokenLength(section_text, tokenizer) > 4096):
                # print(f"🔹 Original text: {section_text}")
                isMultiChunk = True

            if isMultiChunk and i%400 == 1:
                print(f"Original text: {section_text}")
                print(f"🔹 Original text token count: {isExceedsTokenLength(section_text, tokenizer)}")
                print(f"🔸 Number of chunks: {len(chunks)}")


            for chunk in chunks:
                token_len = isExceedsTokenLength(chunk, tokenizer)
                if isMultiChunk and i%400 == 1:
                    print(f"Chunck: {chunk}")
                    print(f"  ⤷ Chunk token length: {token_len}")
                
                chunk = preprocess_vector(chunk)
                embedding = model.encode(chunk).astype(np.float32)


                document = {
                    "Court_Name": str(df["Court Name"].iloc[i]),
                    "Title": str(df["Title"].iloc[i]),
                    "Judge": str(df["Judge"].iloc[i]),
                    "CNR": str(df["CNR"].iloc[i]),
                    "Date_of_Registration": str(df["Date of Registration"].iloc[i]),
                    "Decision_Date": str(df["Decision Date"].iloc[i]),
                    "Disposal_Nature": str(df["Disposal Nature"].iloc[i]),
                    "State": str(df["State"].iloc[i]),
                    "PDF_Path": str(df["PDF Path"].iloc[i]),
                    "Text": chunk,
                    "Case_Number": str(df["Case Number"].iloc[i]),
                    "id": db_id,
                    "Embedding": embedding
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
        "Court_Name",
            "Title",
            "Judge",
            "CNR",
            "Date_of_Registration",
            "Decision_Date",
            "Disposal_Nature",
            "State",
            "PDF_Path",
            "Text",
            "Case_Number",
            "id",
        ]

        try:
            all_records = collection.query(expr="id >= 0", output_fields=output_fields)
            
            # Option 1: Print to console
            print(f"Total records in Milvus: {len(all_records)}")
            for rec in all_records[:5]:  # Print first 5 for brevity
                print(rec)

            # Option 2: Save to JSON
            with open(f"milvus_actual_data_{collection_name}.json", "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
                
        except Exception as query_error:
            print(f"⚠️ Warning: Could not query collection: {query_error}")
            print(f"📊 Using JSON snapshot data instead (contains {len(json_records)} records)")
            
            # Use the JSON records we already have as fallback
            with open(f"milvus_actual_data_{collection_name}.json", "w", encoding="utf-8") as f:
                json.dump(json_records, f, ensure_ascii=False, indent=2)

        print(f"✅ Completed indexing for {collection_name}")

    except Exception as e:
        print(f"❌ Error processing {collection_name}: {str(e)}")
