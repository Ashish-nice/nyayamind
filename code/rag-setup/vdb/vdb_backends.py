"""Milvus and Endee backends for the NyayaMind test-set retrieval.

Both strategies are dense-only, as in code/rag-setup/milvus and code/rag-setup/endee:

    Milvus  L2 distance; IVF_FLAT (nlist 2048, nprobe 32) for Central Acts and
            Supreme Court, FLAT (exact) for State Acts and High Courts
    Endee   cosine HNSW, M 16, ef_con 128, ef_search 128

The vectors are the Snowflake arctic-embed-m-v2.0 embeddings already computed
for the Vespa collections (data/vespa_feed shards, bf16), so the three
strategies differ only in the vector database and search method, not in the
embeddings. bf16 values are stored as float16 (exactly representable at the
magnitudes of a unit-norm 768-d vector).

Only ids, vectors and the metadata needed for search/filtering live in the
vector databases. Judgment text (needed for self-exclusion, citation
propagation and the final output) is kept in a local SQLite side store
(data/vdb_textstore.sqlite), so the databases fit the laptop's Docker VM.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vespa"))
import common as c  # noqa: E402

DIM = c.EMBED_DIM
TEXT_STORE = c.DATA_DIR / "vdb_textstore.sqlite"

# collection -> (kind, source schemas)
COLLECTIONS: dict[str, tuple[str, list[str]]] = {
    "centralacts": ("statute", [c.CA_SCHEMA]),
    "stateacts": ("statute", c.sa_schemas()),
    "scjudgments": ("case", [c.SC_SCHEMA]),
    "hcjudgments": ("case", [c.HC_SCHEMA]),
}
KIND_COLLECTIONS = {
    "statute": ["centralacts", "stateacts"],
    "case": ["scjudgments", "hcjudgments"],
}


def decode_bf16(hex_values: str) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(hex_values), dtype=">u2").astype(np.uint32) << 16
    return raw.view(np.float32)


def iter_docs(collection: str) -> Iterator[dict[str, Any]]:
    for schema in COLLECTIONS[collection][1]:
        for doc in c.iter_feed_shards(schema):
            yield doc


# --------------------------------------------------------------------------
# Text side store
# --------------------------------------------------------------------------
class TextStore:
    def __init__(self, path: Path = TEXT_STORE):
        self.path = path
        self.local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self.local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            self.local.conn = conn
        return conn

    def get(self, doc_id: str) -> str:
        row = self._conn().execute("SELECT body FROM docs WHERE id=?", (doc_id,)).fetchone()
        return zlib.decompress(row[0]).decode("utf-8") if row else ""

    def get_many(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        out = {}
        for start in range(0, len(ids), 500):
            part = ids[start: start + 500]
            q = f"SELECT id, body FROM docs WHERE id IN ({','.join('?' * len(part))})"
            for doc_id, body in self._conn().execute(q, part):
                out[doc_id] = zlib.decompress(body).decode("utf-8")
        return out

    @staticmethod
    def exists(path: Path = TEXT_STORE) -> bool:
        return path.exists() and path.with_suffix(".complete").exists()


class TextStoreWriter:
    def __init__(self, path: Path = TEXT_STORE):
        self.path = path
        if path.exists():
            path.unlink()
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, body BLOB)")
        self.buffer: list[tuple[str, bytes]] = []

    def add(self, doc_id: str, text: str) -> None:
        self.buffer.append((doc_id, zlib.compress(text.encode("utf-8"), 3)))
        if len(self.buffer) >= 5000:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.conn.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?)", self.buffer)
            self.conn.commit()
            self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.conn.close()
        self.path.with_suffix(".complete").write_text("ok")


# --------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------
def row_meta(kind: str, doc: dict[str, Any]) -> dict[str, str]:
    f = doc["fields"]
    if kind == "statute":
        return {"act": str(f.get("name_of_statute", ""))[:900]}
    return {
        "case_name": str(f.get("case_name", ""))[:1500],
        "judgment_date": str(f.get("judgment_date", ""))[:32],
    }


# --------------------------------------------------------------------------
# Milvus
# --------------------------------------------------------------------------
class MilvusBackend:
    name = "milvus"
    INDEX = {"centralacts": "IVF_FLAT", "stateacts": "FLAT", "scjudgments": "IVF_FLAT", "hcjudgments": "FLAT"}

    def __init__(self, uri: str | None = None):
        from pymilvus import MilvusClient

        self.uri = uri or os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
        self.client = MilvusClient(uri=self.uri, timeout=600)
        self.lock = threading.Lock()

    def ready(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def count(self, collection: str) -> int:
        if not self.client.has_collection(collection):
            return 0
        stats = self.client.get_collection_stats(collection)
        return int(stats.get("row_count", 0))

    def create(self, collection: str) -> None:
        from pymilvus import DataType

        kind = COLLECTIONS[collection][0]
        if self.client.has_collection(collection):
            self.client.drop_collection(collection)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("embedding", DataType.FLOAT16_VECTOR, dim=DIM)
        if kind == "statute":
            schema.add_field("act", DataType.VARCHAR, max_length=1000)
        else:
            schema.add_field("case_name", DataType.VARCHAR, max_length=2000)
            schema.add_field("judgment_date", DataType.VARCHAR, max_length=64)
        self.client.create_collection(collection, schema=schema)

    def insert(self, collection: str, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        kind = COLLECTIONS[collection][0]
        rows = []
        for doc, vec in zip(docs, vectors.astype(np.float16)):
            rows.append({"id": doc["id"], "embedding": vec, **row_meta(kind, doc)})
        for attempt in range(5):
            try:
                self.client.insert(collection, rows)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"  insert retry ({exc})", flush=True)
                time.sleep(5 * (attempt + 1))

    def finalize(self, collection: str) -> None:
        self.client.flush(collection)
        index = self.INDEX[collection]
        params = self.client.prepare_index_params()
        params.add_index("embedding", index_type=index, metric_type="L2",
                         params={"nlist": 2048} if index == "IVF_FLAT" else {})
        self.client.create_index(collection, params)
        self.load(collection)

    def load(self, collection: str) -> None:
        self.client.load_collection(collection)

    def search(self, collection: str, vector: np.ndarray, limit: int,
               acts: list[str] | None = None) -> list[dict[str, Any]]:
        kind = COLLECTIONS[collection][0]
        index = self.INDEX[collection]
        params = {"metric_type": "L2", "params": {"nprobe": 32} if index == "IVF_FLAT" else {}}
        fields = ["act"] if kind == "statute" else ["case_name", "judgment_date"]
        kwargs: dict[str, Any] = {}
        if acts:
            kwargs["filter"] = "act in {acts}"
            kwargs["filter_params"] = {"acts": [a[:900] for a in acts]}
        res = self.client.search(collection, data=[np.asarray(vector, dtype=np.float16)],
                                 anns_field="embedding", limit=limit, search_params=params,
                                 output_fields=fields, **kwargs)
        hits = []
        for h in res[0]:
            ent = h.get("entity", {})
            # smaller L2 distance = closer; negate so that higher score = better everywhere
            hits.append({"id": h["id"], "score": -float(h["distance"]), "fields": dict(ent)})
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "Milvus standalone v2.5.18 (embedded etcd, local storage)",
            "metric": "L2",
            "vector_type": "FLOAT16_VECTOR (768-d)",
            "indexes": {k: (f"{v} nlist=2048, nprobe=32" if v == "IVF_FLAT" else "FLAT (exact)")
                        for k, v in self.INDEX.items()},
        }


# --------------------------------------------------------------------------
# Endee
# --------------------------------------------------------------------------
class EndeeBackend:
    """Endee server 1.3.5 (endeeio/endee-server image) through the v1 client
    (endee==0.1.27), the same API as code/rag-setup/endee: create_index with
    HNSW M/ef_con, index.upsert, index.query(vector, top_k, ef, filter)."""

    name = "endee"
    M, EF_CON, EF_SEARCH = 16, 128, 128
    # the repo scripts used precision="medium" (16-bit) in the old API; float16 here
    PRECISION = "float16"

    def __init__(self, url: str | None = None):
        from endee import Endee

        self.url = url or os.environ.get("ENDEE_URL", "http://127.0.0.1:8090/api/v1")
        self.client = Endee()
        self.client.set_base_url(self.url)
        self.indexes: dict[str, Any] = {}
        self.lock = threading.Lock()

    def ready(self) -> bool:
        try:
            self.client.list_indexes()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _index(self, collection: str):
        with self.lock:
            if collection not in self.indexes:
                self.indexes[collection] = self.client.get_index(collection)
            return self.indexes[collection]

    def count(self, collection: str) -> int:
        try:
            for info in self.client.list_indexes().get("indexes", []):
                if info.get("name") == collection:
                    return int(info.get("total_elements", 0))
        except Exception:  # noqa: BLE001
            pass
        return 0

    def create(self, collection: str) -> None:
        try:
            self.client.delete_index(collection)
        except Exception:  # noqa: BLE001
            pass
        self.indexes.pop(collection, None)
        self.client.create_index(name=collection, dimension=DIM, space_type="cosine",
                                 M=self.M, ef_con=self.EF_CON, precision=self.PRECISION)

    def insert(self, collection: str, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        kind = COLLECTIONS[collection][0]
        items = []
        for doc, vec in zip(docs, vectors):
            meta = row_meta(kind, doc)
            item = {"id": doc["id"], "vector": vec.astype(np.float32).tolist(), "meta": meta}
            if kind == "statute":
                item["filter"] = {"act": meta["act"]}
            items.append(item)
        for attempt in range(5):
            try:
                self._index(collection).upsert(items)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"  upsert retry ({exc})", flush=True)
                time.sleep(5 * (attempt + 1))

    def finalize(self, collection: str) -> None:
        pass

    def load(self, collection: str) -> None:
        pass

    def search(self, collection: str, vector: np.ndarray, limit: int,
               acts: list[str] | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if acts:
            kwargs["filter"] = [{"act": {"$in": [a[:900] for a in acts]}}]
            kwargs["prefilter_cardinality_threshold"] = 20000
        res = self._index(collection).query(vector=np.asarray(vector, dtype=np.float32).tolist(),
                                            top_k=limit, ef=self.EF_SEARCH, **kwargs)
        hits = []
        for h in res:
            # Server 1.3.5 swaps similarity/distance on filtered queries (order is
            # correct); verified on a toy index, so read the cosine from "distance".
            score = float(h.get("distance", 0.0)) if acts else float(h.get("similarity", 0.0))
            hits.append({"id": h["id"], "score": score, "fields": dict(h.get("meta") or {})})
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "Endee server 1.3.5 (endeeio/endee-server), client endee==0.1.27 (v1 API)",
            "metric": "cosine",
            "vector_type": f"{self.PRECISION} (768-d)",
            "indexes": {k: f"HNSW M={self.M}, ef_con={self.EF_CON}, ef_search={self.EF_SEARCH}"
                        for k in COLLECTIONS},
            "note": "filtered queries: server returns similarity/distance swapped; ranking uses the true cosine",
        }


def make_backend(name: str):
    return {"milvus": MilvusBackend, "endee": EndeeBackend}[name]()
