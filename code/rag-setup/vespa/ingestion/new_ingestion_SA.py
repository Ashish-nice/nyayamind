import pandas as pd
import numpy as np
import os
import unicodedata
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from vespa.application import Vespa
import torch

# --- Configuration ---
VESPA_URL = "http://localhost:8085"
# This list matches the keys/filenames used in your setup script
STATE_LIST = ["Lakshadweep", "Madhya Pradesh", "Tamil Nadu", "Chandigarh", "Bihar",
    "Uttarakhand", "Rajasthan", "Jharkhand", "Jammu and Kashmir", "Odisha",
    "Puducherry", "Punjab", "Andhra Pradesh", "West Bengal", "Tripura",
    "Ladakh", "Chhattisgarh", "Maharashtra", "Telangana", "Himachal Pradesh",
    "Gujarat", "Manipur", "Karnataka", "Arunachal Pradesh",
    "Andaman and Nicobar Islands", "Assam", "Delhi", "Goa", "Meghalaya",
    "Dadra and Nagar Haveli and Daman and Diu", "Kerala", "Haryana",
    "Uttar Pradesh"]
# Example full list:
# STATE_LIST = ['Lakshadweep', 'Madhya Pradesh', 'Tamil Nadu', ... ]

# 1. Load Model
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
print(f"Loading model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Connect to Vespa
app = Vespa(url=VESPA_URL)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = ' '.join(text.split())
    return text

def get_token_length(text):
    tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)
    return len(tokens['input_ids'])

# 3. Setup Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8192,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
    length_function=get_token_length
)

def ingest_data():
    for state_name in STATE_LIST:
        # Construct filepath based on the list item
        file_path = f'/home/shubham/tanmay/lexensemble/code/state_acts_csv/{state_name}.csv'

        if not os.path.exists(file_path):
            print(f"⚠️ Skipping `{state_name}` — CSV not found at {file_path}")
            continue

        print(f"\n🔍 Processing state: {state_name}")
        df = pd.read_csv(file_path)

        # --- KEY CHANGE: Dynamic Schema Name ---
        # Matches new_setup_merged.py logic: "sa_" + lowercase + underscores
        clean_name = state_name.lower().replace(" ", "_")
        schema_name = f"sa_{clean_name}"

        vespa_feed = []
        global_id_counter = 0

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Ingesting into {schema_name}"):
            original_text = row.get('Section Text', '')
            if pd.isna(original_text): continue

            chunks = text_splitter.split_text(str(original_text))

            for chunk in chunks:
                clean_chunk = preprocess_text(chunk)
                embedding = model.encode(clean_chunk).tolist()

                # Unique Doc ID
                doc_id = f"{schema_name}_{row.get('id', i)}_{global_id_counter}"

                vespa_fields = {
                    "id": doc_id,
                    "original_doc_id": int(row.get("id", 0)) if pd.notna(row.get("id")) else 0,
                    "state_name": str(row.get("State Name", state_name)),
                    "name_of_statute": str(row.get("Name of Statute", "")),
                    "section_number": str(row.get("Section Number", "")),
                    "section_title": str(row.get("Section Title", "")),
                    "section_text": clean_chunk,
                    "embedding": embedding
                }

                vespa_feed.append({
                    "id": doc_id,
                    "fields": vespa_fields
                })
                global_id_counter += 1

                # Feed in batches
                if len(vespa_feed) >= 100:
                    app.feed_iterable(iter(vespa_feed), schema=schema_name)
                    vespa_feed = []

        # Flush remaining
        if vespa_feed:
            app.feed_iterable(iter(vespa_feed), schema=schema_name)
        print(f"✅ Completed {state_name}")

if __name__ == "__main__":
    ingest_data()
