# NyayaMind retrieval: Vespa strategy

This is a hybrid retrieval setup over NyayaMindDB, following the NyayaMind paper (arXiv 2604.09069, Section 5). For each query it runs two Vespa searches in parallel, a dense search (top-10, HNSW) and a BM25 search (top-10). The two lists are merged client-side with Reciprocal Rank Fusion (k = 60). There is no cross-encoder re-ranking. Embeddings come from `Snowflake/snowflake-arctic-embed-m-v2.0` (768-d), on chunks of up to 8192 tokens.

| Collection | Vespa schema | Source | Ingestion | Query endpoint |
|---|---|---|---|---|
| Central Acts | `centralacts` | `data/CA/Central_Acts.csv` | `ingestion/ingestion_CA.py` | `retrieval/query_CA.py` → `search_central_acts()` |
| State Acts | `sa_<state>` (one per state) | `data/SA/<State>.csv` | `ingestion/new_ingestion_SA.py` | `retrieval/new_query_SA.py` → `search_state_acts()` |
| Supreme Court | `scjudgments` | `data/SC/SC_data.csv` | `ingestion/create_sc_collections.py` | `retrieval/query_SC.py` → `search_sc_judgments()` |
| High Courts | `hcjudgments` | `data/HC/hc_cases_judgements.csv` | `ingestion/ingestion_HC_new.py` | `retrieval/query_HC_new.py` → `search_hc_judgments()` |

The remaining files:

* `common.py` holds the shared configuration and helpers (endpoint, embedding model, chunking, feed shards, hybrid search).
* `new_setup_merged.py` builds and deploys the application with all schemas.
* `count_documents.py` prints the number of documents in each schema.
* `retrieval/retrieve_test_set.py` runs offline retrieval for `data/test_data/test_v2.json`.

## Running on a laptop (tested: 16 GB RAM, RTX 3060 6 GB, Windows + Docker Desktop)

```bash
pip install -r code/rag-setup/vespa/requirements.txt

# 1. Start Docker Desktop, then deploy Vespa
#    (named volume, 6 GB cap, http://localhost:8085)
python code/rag-setup/vespa/new_setup_merged.py

# 2. Ingest. Each script has two stages: "embed" writes resumable offline shards
#    to data/vespa_feed/ (GPU only, Vespa not needed); "feed" pushes them to Vespa.
python code/rag-setup/vespa/ingestion/ingestion_CA.py          # ~10 min
python code/rag-setup/vespa/ingestion/new_ingestion_SA.py      # ~20 min
python code/rag-setup/vespa/ingestion/create_sc_collections.py # ~2-3 h
python code/rag-setup/vespa/ingestion/ingestion_HC_new.py      # long, see below
python code/rag-setup/vespa/count_documents.py

# 3. Offline retrieval for the test set (pre-processing; no model inference)
python code/rag-setup/vespa/retrieval/retrieve_test_set.py
#    writes data/test_data/retrieval_vespa/{query_embeddings.npy, retrieval.jsonl, retrieval_report.json}
#    and data/test_data/test_v2_rag.json
```

Every stage can be interrupted and re-run. Embedding resumes from the last finished shard, and test-set retrieval skips cases it has already done (`--overwrite` redoes them).

Once the model has been downloaded, you can run fully offline with `HF_HUB_OFFLINE=1`. You can override paths and endpoints with these environment variables: `NYAYAMIND_DATA_DIR`, `NYAYAMIND_FEED_DIR`, `VESPA_URL`, `VESPA_PORT`.

### High Court collection

The CSV is 9.4 GB: 811,493 judgments, about 2.4 B tokens. By default the script reproduces the paper's Vespa HC collection, which has one document per judgment (811,492 in the paper's Table 6). The first 8192 tokens are embedded and the first 60,000 characters are indexed for BM25. Test-set retrieval can run before HC is ready by passing `--skip HC`.

### Laptop profile (finishes in about 8–10 h)

Under sustained load, a laptop RTX 3060 hits its 87 °C thermal target and slows to about a third of its benchmark speed. At that speed the paper's full-length embeddings would take more than 60 h. The laptop profile keeps everything the same except how much of each document the dense vector covers. `--embed-tokens N` embeds only the first N tokens of each stored chunk. BM25 still indexes the full stored text, so the hybrid search keeps full-text lexical coverage.

```bash
V=code/rag-setup/vespa/ingestion
# GPU: embed SC, then HC (writes shards)
python $V/create_sc_collections.py --stage embed --embed-tokens 2048 && \
python $V/ingestion_HC_new.py      --stage embed --embed-tokens 512
# In a second terminal, at the same time: feed shards as they appear
python $V/create_sc_collections.py --stage feed --follow && \
python $V/ingestion_HC_new.py      --stage feed --follow
# Then retrieve for test cases 1-300
python code/rag-setup/vespa/retrieval/retrieve_test_set.py --limit 300
```

| Collection | Stored and indexed for BM25 | Dense embedding |
|---|---|---|
| SC | 8192-token chunks (paper setting) | first 2048 tokens of each chunk |
| HC | one document per judgment, first 60k characters (paper setting) | first 512 tokens (court, parties, coram, opening facts) |
| CA, SA | full sections (paper setting) | full sections |

## Test-set retrieval v2 (cases 1-300)

```bash
python code/rag-setup/vespa/retrieval/retrieve_test_set.py --limit 300 --propagate-from 30
# -> data/test_data/retrieval_vespa_v2/{test_v2_rag.json, test_v2_rag_inference.json, retrieval_report.json}
```

v1 took the fused top 5 from one hybrid query over the whole facts text. v2 keeps that query and adds more evidence lists, all fused with weighted RRF:

* **Explicit references.** Phrases such as "Section 3 of the 1971 Act" or "Sections 279, 337 IPC" in the facts are resolved against a catalogue of every corpus section. The resolver code is in `retrieval/statute_refs.py`.
* **Per-passage queries.** Long facts dilute a single dense vector, so each passage is also searched on its own.
* **Named-Act search.** Hybrid search is restricted to the Acts named in the facts.
* **Citation propagation.** The 30 most similar judgments are mined for the sections and the "X v. Y" precedents they cite. Those citations are resolved against a catalogue of the 849k corpus judgments (`retrieval/case_refs.py`).
* **Temporal cut-off.** Nothing decided on or after the judgment's own date is retrieved, and no Act enacted after it.

The outputs:

* `test_v2_rag.json` is structured as `{meta, cases: [{case, query, retrieved: {statutes, precedents}, gold, evaluation}]}`. Each retrieved item carries its title, metadata, text, score and the evidence lists that produced it.
* `test_v2_rag_inference.json` has the flat `test_v2.json` layout that the inference scripts expect.

The fusion weights are hand-set (`DEFAULT_WEIGHTS`). `--stage merge --weights <json>` re-fuses the results without querying Vespa again. v1 results are kept in `data/test_data/retrieval_vespa_v1/`.

## What the test-set retrieval uses as input

The paper says: *"At inference time, NyayaMind takes as input the factual description of a legal case."* So the query is each case's `Simplified_Facts`, the field the prediction prompts call "Case proceedings" (`Facts` is used only as a fallback). Issues, arguments, reasoning, the decision and the gold citations are never used to retrieve.

The case's own judgment is removed from the SC and HC results. A result is treated as the case's own judgment when its party names match the case or it contains verbatim passages from the case's facts.

`test_v2_rag.json` keeps the layout of `test_v2.json`. The retrieved statutes and precedents go into `section_titles`, `section_texts`, `cited_cases` and `cited_case_judgments`. The gold values are kept as `gold_section_titles` and `gold_cited_cases`, and per-collection hits and scores are under `retrieval`. Because the layout is unchanged, the inference scripts run on the retrieved context with no code changes:

```bash
python code/inference/inference_Claude_Opus_5_5.py --dataset data/test_data/test_v2_rag.json
```

## Changes relative to the original scripts

* **Python 3.13 and transformers 5 support.** The model's remote code needs xformers and relies on buffers that transformers 5 leaves uninitialised. `common.load_embedder()` switches to PyTorch SDPA attention and rebuilds those buffers. The GPU fp16 output was checked against the CPU fp32 output (cosine 1.000) and for batch-padding invariance.
* **Query prompt.** Queries use arctic-embed's `query: ` prompt, as the model card requires. Documents use no prompt.
* **Statute embeddings.** Statute chunks are embedded with an `Act | Section n. Title` header, so a section whose body never names its Act can still be found. Text keeps its original case: Vespa's BM25 is case-insensitive anyway, and the text is meant to be shown to an LLM.
* **Memory and storage.** Vectors are stored as `bfloat16`. Vespa data lives in a named Docker volume instead of a Windows bind mount.
* **BM25 queries.** These run as `weakAnd` over cleaned query terms (at most 96). Under the default `all` mode, a long facts query matches nothing, and raw text can break Vespa's query syntax.
* **Configuration fixes.** All scripts now use one endpoint and the correct schema names. The originals mixed ports 8083 and 8085, queried a non-existent `HC_Judgments_DB`, and queried placeholder `sa_sample_part*` schemas. The SC ingestion script, which was empty, is now implemented.
