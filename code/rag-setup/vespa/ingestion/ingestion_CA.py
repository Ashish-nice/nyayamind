import pandas as pd
import numpy as np
import unicodedata
import json
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from tqdm import tqdm
from vespa.application import Vespa
import torch

# --- Configuration ---
CSV_PATH = '/home/shubham/tanmay/lexensemble/data/CA/Central_Acts.csv'
VESPA_URL = "http://localhost:8085"
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
MAX_TOKENS = 8192

# Initialize Vespa connection
app = Vespa(url=VESPA_URL)

# Load Model & Tokenizer
print(f"🔄 Loading model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    return ' '.join(text.split())

def count_tokens(text):
    # Returns the number of tokens in the text
    tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)
    return len(tokens['input_ids'])

# Text Splitter with 8192 token limit
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_TOKENS,
    chunk_overlap=100,
    separators=["\n\n", "\n \n", ". "],
    length_function=count_tokens
)

def feed_data():
    df = pd.read_csv(CSV_PATH)
    vespa_feed = []

    print(f"📂 Reading {len(df)} records from CSV...")

    # Iterate using iterrows to easily access row data
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing & Ingesting"):
        section_text = row.get('Section Text', '')

        if pd.isna(section_text):
            continue

        # Split text based on new 8192 token limit
        chunks = text_splitter.split_text(str(section_text))

        for chunk_idx, chunk in enumerate(chunks):
            clean_chunk = preprocess_text(chunk)

            # Encode
            embedding = model.encode(clean_chunk).tolist()

            # Create a unique ID for the chunk
            doc_id = f"{i}_{chunk_idx}"

            vespa_fields = {
                "id": doc_id,
                "section_text": clean_chunk,
                "name_of_statute": str(row.get('Name of statute', '')),
                "section_number": str(row.get('Section Number', '')),
                "section_title": str(row.get('Section Title', '')),
                "embedding": embedding
            }

            # Add to batch list
            vespa_feed.append({
                "id": doc_id,
                "fields": vespa_fields
            })

            # Push in batches of 100 to avoid memory overflow
            if len(vespa_feed) >= 100:
                app.feed_iterable(iter(vespa_feed), schema="centralacts")
                vespa_feed = []

    # Flush remaining documents
    if vespa_feed:
        app.feed_iterable(iter(vespa_feed), schema="centralacts")

if __name__ == "__main__":
    feed_data()
    print("✅ Ingestion Complete.")
