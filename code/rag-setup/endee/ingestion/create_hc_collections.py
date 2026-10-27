import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import os
import unicodedata
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
    handlers=[logging.FileHandler('vectorx_hc_processing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

script_start_time = time.time()
print(f"🚀 Starting script at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("📦 Loading model...")
model_load_start = time.time()
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", device="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-v2.0")
embedding_dim = model.get_sentence_embedding_dimension()
model_load_time = time.time() - model_load_start
print(f"✅ Model loaded in {model_load_time:.2f} seconds")

def safe_insert_data(index, data: List[Dict], max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Inserting {len(data)} records (attempt {attempt + 1}/{max_retries})")
            response = index.upsert(data)
            logger.info(f"✅ Successfully inserted batch. Response: {response}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Insert attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    logger.error("❌ Failed to insert data after retries")
    return False

def preprocess_vector(text):
    if pd.isna(text):
        return ""
    text = text.replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return ' '.join(text.lower().split())

def isExceedsTokenLength(text, tokenizer, max_length=4096):
    if not text: return 0
    try:
        return len(tokenizer(text, return_attention_mask=False, return_token_type_ids=False)['input_ids'])
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

# List of High Court CSVs and their collection names
file_paths = [
    # Add your CSV file paths here
    "/home/shubham/tanmay/lexensemble/code/hc_cases_judgements.csv",
    # ... add more as needed ...
]
collection_map = {
    "/home/shubham/tanmay/lexensemble/code/hc_cases_judgements.csv": "HC_Judgments_DB",
    # ... add more as needed ...
}

text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4096,
                    chunk_overlap=100,
                    separators=["\n\n", "\n \n", r"\n\d+\. ", ". "],
                    length_function=lambda x: isExceedsTokenLength(x, tokenizer)
                )

# Main processing
for file_path in file_paths:
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ Skipping file not found: {file_path}")
        continue
    try:
        collection_name = collection_map[file_path]
        logger.info(f"\n🔍 Processing state: {collection_name}")
        create_endee_index(collection_name, embedding_dim)
        collection = client.get_index(collection_name)

        df = pd.read_csv(file_path)
        db_id = 0
        batch_size = 1000
        skipped_rows = 0
        skipped_chunks = 0
        texts_to_embed = []
        metadata_buffer = deque()
        MAX_TOKEN_COUNT = 32000

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                section_text = str(row['Text']) if 'Text' in row and not pd.isna(row['Text']) else ""
                if not section_text:
                    skipped_rows += 1
                    continue
                token_count = isExceedsTokenLength(section_text, tokenizer)
                if token_count > MAX_TOKEN_COUNT:
                    skipped_rows += 1
                    logger.warning(f"Skipping row {i+1}: Extremely large content ({token_count} tokens)")
                    continue
                chunks =  text_splitter.split_text(section_text)
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
                        "original_doc_id": int(row.get('original_row_number', i)),
                        "Text": clean_chunk[:60000],
                        "Court_Name": str(row.get('Court Name', ''))[:4000],
                        "Title": str(row.get('Title', ''))[:4000],
                        "Judge": str(row.get('Judge', ''))[:4000],
                        "CNR": str(row.get('CNR', ''))[:4000],
                        "Date_of_Registration": str(row.get('Date of Registration', ''))[:4000],
                        "Decision_Date": str(row.get('Decision Date', ''))[:4000],
                        "Disposal_Nature": str(row.get('Disposal Nature', ''))[:4000],
                        "State": str(row.get('State', ''))[:4000],
                        "PDF_Path": str(row.get('PDF Path', ''))[:4000],
                        "Case_Number": str(row.get('Case Number', ''))[:4000],
                    })
                    db_id += 1
                    if len(texts_to_embed) >= batch_size:
                        try:
                            embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=False).astype(np.float32)
                            batch_data = []
                            for m, e in zip(metadata_buffer, embeddings):
                                metadata = {k: v for k, v in m.items()}
                                batch_data.append({
                                    "id": m["id"],
                                    "vector": e.tolist(),
                                    "meta": metadata,
                                    "filter": {
                                        "Court_Name": m["Court_Name"],
                                        "Case_Number": m["Case_Number"],
                                        "State": m["State"],
                                        "original_doc_id": str(m["original_doc_id"])
                                    }
                                })
                            safe_insert_data(collection, batch_data)
                        except Exception as e:
                            logger.error(f"❌ Error encoding batch: {e}")
                        texts_to_embed.clear()
                        metadata_buffer.clear()
            except Exception as e:
                logger.error(f"❌ Error on row {i+1}: {e}")
                skipped_rows += 1

        if texts_to_embed:
            try:
                embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=False).astype(np.float32)
                batch_data = []
                for m, e in zip(metadata_buffer, embeddings):
                    metadata = {k: v for k, v in m.items()}
                    batch_data.append({
                        "id": m["id"],
                        "vector": e.tolist(),
                        "meta": metadata,
                        "filter": {
                            "Court_Name": m["Court_Name"],
                            "Case_Number": m["Case_Number"],
                            "State": m["State"],
                            "original_doc_id": str(m["original_doc_id"])
                        }
                    })
                safe_insert_data(collection, batch_data)
            except Exception as e:
                logger.error(f"❌ Error encoding final batch: {e}")
        logger.info(f"\n📊 State: {collection_name}")
        logger.info(f"   Processed: {len(df)} rows")
        logger.info(f"   Skipped rows: {skipped_rows}")
        logger.info(f"   Skipped chunks: {skipped_chunks}")
        logger.info(f"   Indexed documents: {db_id}")
        logger.info(f"✅ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")

total_script_time = time.time() - script_start_time
print(f"\n🏁 All high court collections processed")
print(f"⏱️ Total script time: {total_script_time:.2f}s")
print(f"✅ Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
