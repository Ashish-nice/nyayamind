import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import os
import unicodedata
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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("state_acts_indexing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

script_start_time = time.time()
print(f"🚀 Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("📦 Loading embedding model...")
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

print(f"✅ Model loaded in {time.time() - model_load_start:.2f}s")

def safe_insert_data(index, data: List[Dict], max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Inserting {len(data)} vectors (attempt {attempt + 1})")
            index.upsert(data)
            logger.info("✅ Insert successful")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Insert failed: {e}")
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    logger.error("❌ Insert failed after retries")
    return False


def preprocess_vector(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.replace("\n", " ")
    text = unicodedata.normalize("NFKD", text)\
        .encode("ascii", "ignore")\
        .decode("utf-8")
    return " ".join(text.lower().split())


def token_length(text: str) -> int:
    if not text:
        return 0
    try:
        return len(
            tokenizer(
                text,
                return_attention_mask=False,
                return_token_type_ids=False
            )["input_ids"]
        )
    except Exception:
        return 0

def create_endee_index(name: str, dim: int):
    client.create_index(
        name=name,
        dimension=dim,
        space_type="cosine",
        M=16,
        ef_con=128,
        precision="float32"
    )

collection_name = "State_Act_v2"
batch_size = 1000
MAX_TOKEN_COUNT = 32000

state_list = [
    "Lakshadweep", "Madhya Pradesh", "Tamil Nadu", "Chandigarh", "Bihar",
    "Uttarakhand", "Rajasthan", "Jharkhand", "Jammu and Kashmir", "Odisha",
    "Puducherry", "Punjab", "Andhra Pradesh", "West Bengal", "Tripura",
    "Ladakh", "Chhattisgarh", "Maharashtra", "Telangana", "Himachal Pradesh",
    "Gujarat", "Manipur", "Karnataka", "Arunachal Pradesh",
    "Andaman and Nicobar Islands", "Assam", "Delhi", "Goa", "Meghalaya",
    "Dadra and Nagar Haveli and Daman and Diu", "Kerala", "Haryana",
    "Uttar Pradesh"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,
    chunk_overlap=100,
    separators=["\n\n", "\n \n", r"\n\d+\. ", ". "],
    length_function=token_length
)

logger.info("🏗️ Creating index...")
create_endee_index(collection_name, embedding_dim)
index = client.get_index(collection_name)

texts_to_embed = []
metadata_buffer = deque()

db_id = 0
skipped_rows = 0
skipped_chunks = 0

processing_start = time.time()

for state_name in state_list:
    try:
        csv_path = f"/home/shubham/tanmay/lexensemble/code/state_acts_csv/{state_name}.csv"

        if not os.path.exists(csv_path):
            logger.warning(f"⚠️ Missing CSV: {state_name}")
            continue

        logger.info(f"📊 Processing {state_name}")
        df = pd.read_csv(csv_path, header=None)

        df = df.rename(columns={
            0: "id",
            1: "State Name",
            2: "Name of Statute",
            3: "Section Number",
            4: "Section Title",
            5: "Section Text"
        })

        for _, row in tqdm(df.iterrows(), total=len(df)):
            section_text = str(row.get("Section Text", ""))

            if not section_text:
                skipped_rows += 1
                continue

            if token_length(section_text) > MAX_TOKEN_COUNT:
                skipped_rows += 1
                continue

            chunks = text_splitter.split_text(section_text)

            for chunk in chunks:
                clean_chunk = preprocess_vector(chunk)

                if token_length(clean_chunk) > MAX_TOKEN_COUNT:
                    skipped_chunks += 1
                    continue

                texts_to_embed.append(clean_chunk)

                metadata_buffer.append({
                    "id": str(db_id),
                    "original_doc_id": str(row.get("id")),
                    "Text": clean_chunk[:60000],
                    "State_Name": str(row.get("State Name", state_name)),
                    "Name_of_Statute": str(row.get("Name of Statute", "")),
                    "Section_Number": str(row.get("Section Number", "")),
                    "Section_Title": str(row.get("Section Title", ""))
                })

                db_id += 1

                if len(texts_to_embed) >= batch_size:
                    embeddings = model.encode(
                        texts_to_embed,
                        batch_size=32,
                        show_progress_bar=False
                    ).astype(np.float32)

                    batch_data = []
                    for m, e in zip(metadata_buffer, embeddings):
                        batch_data.append({
                            "id": m["id"],
                            "vector": e.tolist(),
                            "meta": m,
                            "filter": {
                                "State_Name": m["State_Name"],
                                "Name_of_Statute": m["Name_of_Statute"],
                                "Section_Number": m["Section_Number"],
                                "original_doc_id": m["original_doc_id"]
                            }
                        })

                    safe_insert_data(index, batch_data)
                    texts_to_embed.clear()
                    metadata_buffer.clear()

    except Exception as e:
        logger.error(f"❌ Error processing {state_name}: {e}")

if texts_to_embed:
    logger.info(f"🔚 Processing final batch ({len(texts_to_embed)} vectors)")

    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=False
    ).astype(np.float32)

    batch_data = []
    for m, e in zip(metadata_buffer, embeddings):
        batch_data.append({
            "id": m["id"],
            "vector": e.tolist(),
            "meta": m,
            "filter": {
                "State_Name": m["State_Name"],
                "Name_of_Statute": m["Name_of_Statute"],
                "Section_Number": m["Section_Number"],
                "original_doc_id": m["original_doc_id"]
            }
        })

    safe_insert_data(index, batch_data)

    texts_to_embed.clear()
    metadata_buffer.clear()

print("\n🏁 Indexing completed")
print(f"📦 Total vectors indexed: {db_id}")
print(f"⚠️ Skipped rows: {skipped_rows}")
print(f"⚠️ Skipped chunks: {skipped_chunks}")
print(f"⏱️ Total time: {time.time() - script_start_time:.2f}s")
print(f"✅ Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")