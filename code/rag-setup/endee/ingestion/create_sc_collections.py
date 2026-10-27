import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from endee import Endee   

client = Endee(token="shubham:NXdEaTCJQEmqnPTKicXNVDFRRvuryuAR")
client.set_base_url("http://172.27.21.35:8080/api/v1")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('vectorx_processing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

script_start_time = time.time()
print(f"🚀 Starting script at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("📦 Loading model...")
model_load_start = time.time()

model = SentenceTransformer(
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    device="cuda",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Snowflake/snowflake-arctic-embed-m-v2.0"
)

embedding_dim = model.get_sentence_embedding_dimension()

print(f"✅ Model loaded in {time.time() - model_load_start:.2f} seconds")

def safe_insert_data(index, data: List[Dict], max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Inserting {len(data)} records (attempt {attempt + 1}/{max_retries})")
            index.upsert(data)
            logger.info("✅ Successfully inserted batch")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Insert attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    logger.error("❌ Failed to insert data after retries")
    return False


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


def isExceedsTokenLength(text, tokenizer, max_length=4096):
    if not text:
        return 0
    try:
        return len(
            tokenizer(
                text,
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']
        )
    except:
        return 0


def create_endee_index(name, embedding_dim):
    client.create_index(
        name=name,
        dimension=embedding_dim,
        space_type="cosine",
        M=16,
        ef_con=128,
        precision="medium"  
    )


print("🏗️ Creating index...")
create_endee_index("SC_Judgments_DB", embedding_dim)

index = client.get_index("SC_Judgments_DB")

print("📊 Loading CSV data...")
df = pd.read_csv('/home/shubham/tanmay/lexensemble/data/SC_data/cleaned/SC_data.csv')
print(f"✅ Loaded {len(df)} rows")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,
    chunk_overlap=100,
    separators=["\n\n", "\n \n", r"\n\d+\. ", ". "],
    length_function=lambda x: isExceedsTokenLength(x, tokenizer)
)

print("🔄 Starting processing and embedding...")
db_id = 0
batch_size = 1000
skipped_rows = 0
skipped_chunks = 0

texts_to_embed = []
metadata_buffer = deque()

MAX_TOKEN_COUNT = 32000

for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
    try:
        file_name = str(row.file_name)
        text_chunks = safe_parse_content(row.content)

        if not isinstance(text_chunks, dict)  or not all(isinstance(v, str) for v in text_chunks.values()):
            skipped_rows += 1
            logger.warning(f"Skipping row {i+1}: Invalid content structure")
            continue

        main_content = text_chunks.get("0", "")
        if isExceedsTokenLength(main_content, tokenizer) > 100000:
            skipped_rows += 1
            continue

        chunks = text_splitter.split_text(main_content)

        for chunk in chunks:
            clean_chunk = preprocess_vector(chunk)
            token_count = isExceedsTokenLength(clean_chunk, tokenizer)
            if token_count > MAX_TOKEN_COUNT:
                skipped_chunks += 1
                logger.warning(f"Skipping chunk with {token_count} tokens (exceeds {MAX_TOKEN_COUNT})")
                continue

            texts_to_embed.append(clean_chunk)

            metadata_buffer.append({
                "id": str(db_id),
                "original_doc_id": str(df["id"].iloc[i]),
                "Text": clean_chunk[:60000],
                "Case_Name": file_name[:20000],
                "Diary_Number": str(row.diary_no),
                "Judgement_Type": str(row.Judgement_type),
                "Case_Number": str(row.case_no),
                "Petitioner": str(row.pet),
                "Respondent": str(row.res),
                "Petitioner_Advocate": str(row.pet_adv),
                "Respondent_Advocate": str(row.res_adv),
                "Bench": str(row.bench),
                "Judgment_date": str(row.judgment_dates),
                "Citations": str(row.citation)
            })

            db_id += 1

            if len(texts_to_embed) >= batch_size:
                embeddings = model.encode(texts_to_embed, batch_size=32).astype(np.float32)

                batch_data = []
                for m, e in zip(metadata_buffer, embeddings):
                    batch_data.append({
                        "id": m["id"],
                        "vector": e.tolist(),
                        "meta": m,
                        "filter": {
                            "Case_Number": m["Case_Number"],
                            "Diary_Number": m["Diary_Number"],
                            "Judgment_date": m["Judgment_date"],
                            "Bench": m["Bench"],
                            "Case_Name": m["Case_Name"],
                            "Judgement_Type": m["Judgement_Type"],
                            "original_doc_id": m["original_doc_id"]
                        }
                    })

                safe_insert_data(index, batch_data)
                texts_to_embed.clear()
                metadata_buffer.clear()

    except Exception as e:
        logger.error(f"❌ Error on row {i}: {e}")
        skipped_rows += 1

if texts_to_embed:
    embeddings = model.encode(texts_to_embed, batch_size=32).astype(np.float32)
    batch_data = []
    for m, e in zip(metadata_buffer, embeddings):
        batch_data.append({
            "id": m["id"],
            "vector": e.tolist(),
            "meta": m,
            "filter": {
                "Case_Number": m["Case_Number"],
                "Diary_Number": m["Diary_Number"],
                "Judgment_date": m["Judgment_date"],
                "Bench": m["Bench"],
                "Case_Name": m["Case_Name"],
                "Judgement_Type": m["Judgement_Type"],
                "original_doc_id": m["original_doc_id"]
            }
        })
    safe_insert_data(index, batch_data)

print(f"\n📊 Processed rows: {len(df)}")
print(f"Skipped rows: {skipped_rows}")
print(f"Skipped chunks: {skipped_chunks}")
print(f"Indexed vectors: {db_id}")
print(f"⏱️ Total time: {time.time() - script_start_time:.2f}s")
print(f"✅ Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
