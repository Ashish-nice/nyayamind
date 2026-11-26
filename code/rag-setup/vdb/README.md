# Milvus and Endee retrieval (test cases 1-300)

Runs the two other NyayaMind retrieval strategies locally on the same embeddings
as the Vespa collections (`data/vespa_feed` shards), so the strategies differ
only in the vector database and search method.

| | Milvus | Endee |
|---|---|---|
| Server | `milvusdb/milvus:v2.5.18`, embedded etcd + local storage, mmap on | `endeeio/endee-server` 1.3.5, client `endee==0.1.27` |
| Metric | L2 | cosine |
| Index | IVF_FLAT (nlist 2048, nprobe 32) for Central Acts and SC; FLAT for State Acts and HC | HNSW M 16, ef_con 128, ef 128, float16 |

```bash
bash code/rag-setup/vdb/containers.sh start-milvus      # or start-endee (localhost only)
python code/rag-setup/vdb/ingest_vdb.py --backend milvus          # ~20 min; also builds data/vdb_textstore.sqlite
python code/rag-setup/vdb/retrieve_test_set_vdb.py --backend milvus   # ~40 min (Endee ~3 min)
python code/rag-setup/vdb/compare_strategies.py         # -> data/test_data/retrieval_comparison.md
```

Outputs per backend in `data/test_data/retrieval_<backend>/`:

* `test_v2_rag.json`, `test_v2_rag_inference.json`, `retrieval_report.json`: the Vespa v2
  pipeline (explicit references, per-passage and named-Act queries, citation propagation,
  weighted RRF, self-exclusion, temporal cut-off) with dense-only search on this engine.
* `baseline/`: the repo scripts' strategy, one dense query over the whole facts, top-10.
* `data/test_data/test_v2_rag_<backend>.json`: copy of the v2 inference file (test_v2 layout).

Notes:

* Vectors are stored as float16; the source embeddings are bf16, so this loses nothing.
* Judgment text lives in a SQLite side store, not in the vector databases.
* Endee 1.3.5 returns `similarity` and `distance` swapped on filtered queries, although the
  order is correct. The backend reads the true cosine.
* Milvus must run with mmap (`docker/user.yaml`). Without it, loading the 811k HC vectors
  ran the 7.9 GB Docker VM out of memory.
