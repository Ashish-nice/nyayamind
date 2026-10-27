

import pandas as pd
import ast
import unicodedata
import re

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from vespa.application import Vespa
import torch


# Connect to the local Vespa instance
app = Vespa(url="http://localhost:8085")

# 1. Load Model
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
print(f"Loading model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
model.max_seq_length = 8192

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    return ' '.join(text.split())

def clean_and_parse_text(raw_text):
    """
    Robust parser that handles messy dictionary strings from CSV.
    Fixes smart quotes, leading zeros, and truncated strings.
    """
    if pd.isna(raw_text):
        return {}

    s = str(raw_text).strip()

    # Attempt 1: Standard Parse
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        pass

    # Attempt 2: Clean Common Syntax Errors
    # Fix smart quotes and ellipses
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("…", "...")

    # Fix "leading zeros in decimal" error (e.g., Act 05 -> Act 5)
    # Regex finds zeros preceded by non-digits and followed by digits
    s = re.sub(r'(?<=\D)0+(\d+)', r'\1', s)
    s = re.sub(r'^0+(\d+)', r'\1', s)

    try:
        return ast.literal_eval(s)
    except Exception:
        # Attempt 3: Fallback
        # If parsing as a dict fails completely (e.g., truncated string),
        # treat the entire raw content as a single chunk rather than discarding it.
        # We strip the outer brackets if they exist to clean it up slightly.
        cleaner_s = s.strip("{}")
        return {"fallback_chunk": cleaner_s}

def feed_data(csv_path, collection_name):
    """
    Ingests data from a CSV into a specific Vespa collection (schema).
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}. Skipping.")
        return

    print(f"\nStarting ingestion for Collection: {collection_name}")
    print(f"Source: {csv_path}")

    df = pd.read_csv(csv_path)
    batch_data = []

    for i in tqdm(range(len(df)), desc=f"Processing {collection_name}"):
        try:
            # --- USE ROBUST PARSER HERE ---
            text_chunks = clean_and_parse_text(df['Text'].iloc[i])

            if not text_chunks:
                continue

            # Metadata
            case_name = str(df['Title'].iloc[i])[:20000]
            date = str(df['Decision Date'].iloc[i])[:2000]
            citation = str(df['Case Number'].iloc[i])[:20000]
            bench = str(df['Judge'].iloc[i])[:20000]

            for chunk_id, chunk_text in text_chunks.items():
                clean_text = preprocess_text(chunk_text)
                if not clean_text:
                    continue

                embedding = model.encode(clean_text).tolist()

                # Correct Vespa ID format using the Dynamic Collection Name
                unique_suffix = f"{i}_{chunk_id}"
                # Format: id:namespace:schema::unique_id
                doc_id = f"id:{collection_name}:{collection_name}::{unique_suffix}"

                vespa_fields = {
                    "id": doc_id,
                    "case_name": case_name,
                    "judgment_date": date,
                    "citations": citation,
                    "bench": bench,
                    "text": clean_text[:60000],
                    "embedding": embedding
                }

                batch_data.append({
                    "id": doc_id,
                    "fields": vespa_fields
                })

                if len(batch_data) >= 100:
                    app.feed_iterable(
                        iter=batch_data,
                        schema=collection_name,     # Dynamic Schema
                        namespace=collection_name,  # Dynamic Namespace
                        callback=lambda response, id: None
                    )
                    batch_data = []

        except Exception as e:
            # Now strictly for unexpected errors, not syntax ones
            print(f"Error processing row {i}: {e}")
            continue

    # Feed remaining items
    if batch_data:
        app.feed_iterable(iter=batch_data, schema=collection_name, namespace=collection_name)
    print(f"Ingestion complete for {collection_name}.")

if __name__ == "__main__":

    # Mapping of CSV file paths to Vespa Collection Names (Schemas)
    collection_map = {"/home/shubham/tanmay/lexensemble/code/hc_cases_judgements.csv": "hcjudgments"}

    # Iterate through the dictionary and process each High Court
    for file_path, collection_name in collection_map.items():
        try:
            feed_data(file_path, collection_name)
        except Exception as main_e:
            print(f"CRITICAL FAILURE processing {collection_name}: {main_e}")
            continue
