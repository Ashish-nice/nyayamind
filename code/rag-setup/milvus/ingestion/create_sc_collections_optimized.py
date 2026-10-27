import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
import pymilvus
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType, Collection
)
import time
import os
import unicodedata
import ast
import json
from transformers import AutoTokenizer
from datetime import datetime
import logging
from typing import List, Dict
from tqdm import tqdm
from collections import deque
import random

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('milvus_processing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

script_start_time = time.time()
print(f"🚀 Starting script at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def connect_to_milvus(max_retries=5, retry_delay=10):
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Attempting to connect to Milvus (attempt {attempt + 1}/{max_retries})")
            connection = connections.connect(host="localhost", port=19530)
            mc = MilvusClient(connections=connection)
            logger.info("✅ Successfully connected to Milvus")
            return connection, mc
        except Exception as e:
            logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay * (2 ** attempt))
    logger.error("❌ Failed to connect to Milvus after all retries")
    raise

def safe_insert_data(collection, data: List[Dict], max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Inserting {len(data)} records (attempt {attempt + 1}/{max_retries})")
            collection.insert(data)
            logger.info(f"✅ Successfully inserted batch")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Insert attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    logger.error("❌ Failed to insert data after retries")
    return False

def check_milvus_health():
    try:
        utility.list_collections()
        return True
    except Exception as e:
        logger.error(f"❌ Milvus health check failed: {e}")
        return False

connection, mc = connect_to_milvus()

# Load model
print("📦 Loading model...")
model_load_start = time.time()
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", device="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-v2.0")
embedding_dim = model.get_sentence_embedding_dimension()
model_load_time = time.time() - model_load_start
print(f"✅ Model loaded in {model_load_time:.2f} seconds")

def preprocess_vector(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return ' '.join(text.lower().split())

def safe_parse_content(content_str):
    if pd.isna(content_str) or content_str == '':
        return {}
    try:
        parsed = ast.literal_eval(str(content_str))
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, (list, tuple)):
            return {str(i): str(v) for i, v in enumerate(parsed)}
        return {"0": str(parsed)}
    except Exception:
        return {"0": str(content_str)}

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='Embedding', dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="Text", dtype=DataType.VARCHAR, max_length=60000),
        FieldSchema(name="Case_Name", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="Diary_Number", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Judgement_Type", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Case_Number", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Petitioner", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Respondent", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Petitioner_Advocate", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Respondent_Advocate", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Bench", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="Judgment_date", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="Citations", dtype=DataType.VARCHAR, max_length=15000),
        FieldSchema(name="original_doc_id", dtype=DataType.INT64),
    ]
    collection = Collection(name=collection_name, schema=CollectionSchema(fields=fields))
    collection.create_index(field_name="Embedding", index_params={"metric_type":"L2", "index_type":"IVF_FLAT", "params":{"nlist":2048}})
    print(f"✅ Collection `{collection_name}` created")
    return collection

def isExceedsTokenLength(text, tokenizer, max_length=4096):
    if not text: return 0
    try:
        return len(tokenizer(text, return_attention_mask=False, return_token_type_ids=False)['input_ids'])
    except:
        return 0

print("🏗️ Creating collection...")
collection_start = time.time()
collection = create_milvus_collection("SC_Judgments_DB", embedding_dim)
collection_create_time = time.time() - collection_start

print("📊 Loading CSV data...")
data_load_start = time.time()
df = pd.read_csv('/home/shubham/tanmay/lexensemble/data/SC_data.csv')
data_load_time = time.time() - data_load_start
print(f"✅ Loaded {len(df)} rows in {data_load_time:.2f}s")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,
    chunk_overlap=100,
    separators=["\n\n", "\n \n", r"\n\d+\. ", ". "],
    length_function=lambda x: isExceedsTokenLength(x, tokenizer)
)

print("🔄 Starting processing and embedding...")
processing_start_time = time.time()
db_id = 0
batch_size = 2000
skipped_rows = 0
texts_to_embed = []
metadata_buffer = deque()

if not check_milvus_health():
    exit("❌ Milvus service is not healthy")

for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
    try:
        file_name = str(row.file_name)
        text_chunks = safe_parse_content(row.content)
        if not isinstance(text_chunks, dict) or not all(isinstance(v, str) for v in text_chunks.values()):
            skipped_rows += 1
            continue
        chunks = text_splitter.split_text(text_chunks.get("0", ""))
        for chunk in chunks:
            clean_chunk = preprocess_vector(chunk)
            texts_to_embed.append(clean_chunk)
            metadata_buffer.append({
                "id": db_id,
                "original_doc_id": int(df["id"].iloc[i]),
                "Text": clean_chunk[:60000],
                "Case_Name": file_name[:20000],
                "Diary_Number": str(row.diary_no)[:2000],
                "Judgement_Type": str(row.Judgement_type)[:2000],
                "Case_Number": str(row.case_no)[:2000],
                "Petitioner": str(row.pet)[:2000],
                "Respondent": str(row.res)[:2000],
                "Petitioner_Advocate": str(row.pet_adv)[:2000],
                "Respondent_Advocate": str(row.res_adv)[:2000],
                "Bench": str(row.bench)[:20000],
                "Judgment_date": str(row.judgment_dates)[:2000],
                "Citations": str(row.citation)[:15000]
            })
            db_id += 1

            if len(texts_to_embed) >= batch_size:
                embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=False).astype(np.float32)
                batch_data = [dict(m, Embedding=e) for m, e in zip(metadata_buffer, embeddings)]
                safe_insert_data(collection, batch_data)
                texts_to_embed.clear()
                metadata_buffer.clear()
    except Exception as e:
        logger.error(f"❌ Error on row {i+1}: {e}")
        skipped_rows += 1

# Final batch
if texts_to_embed:
    embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=False).astype(np.float32)
    batch_data = [dict(m, Embedding=e) for m, e in zip(metadata_buffer, embeddings)]
    safe_insert_data(collection, batch_data)

collection.flush()
processing_time = time.time() - processing_start_time

# Final output summary
print(f"\n📊 Processed: {len(df)} rows")
print(f"   Skipped: {skipped_rows}")
print(f"   Indexed documents: {db_id}")
print(f"⏱️ Total time: {time.time() - script_start_time:.2f}s")
print(f"✅ Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    connections.disconnect("default")
    print("🔌 Disconnected from Milvus")
except Exception as e:
    logger.error(f"❌ Error disconnecting: {e}")
