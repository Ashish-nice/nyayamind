"""Shared configuration and helpers for the NyayaMind Vespa RAG strategy.

Everything that the setup, ingestion and retrieval scripts have in common lives
here, so that schema names, the Vespa endpoint, the embedding model and the
hybrid (dense + BM25 + RRF) search behave identically everywhere.

The paper's Vespa configuration (arXiv 2604.09069, Sec. 5) is reproduced:

* embedding model  : Snowflake/snowflake-arctic-embed-m-v2.0 (768-d)
* chunking         : up to 8192 tokens per chunk, 100-200 token overlap
* ANN index        : HNSW, euclidean, max_links_per_node=16,
                     neighbors_to_explore_at_insert=200
* retrieval        : top-10 nearestNeighbor + top-10 native BM25 queries,
                     fused client side with Reciprocal Rank Fusion (k=60),
                     no cross-encoder re-ranking

Laptop adaptations (16 GB RAM, 6 GB RTX 3060):

* Embeddings are produced in an offline pre-processing stage and written to
  resumable, gzip-compressed feed shards (``data/vespa_feed/<schema>/``).  The
  Vespa feed is a separate stage, so the GPU and Vespa never need to be busy at
  the same time and an interrupted run resumes where it stopped.
* Vectors are stored as ``bfloat16`` in Vespa (half the memory of float).
* Corpora are streamed row by row; the 9 GB High Court CSV is never loaded.
* The model runs in fp16 with PyTorch SDPA attention (xformers is not needed).
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import struct
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths and endpoints (override any of these with environment variables)
# --------------------------------------------------------------------------
VESPA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VESPA_DIR.parents[2]
DATA_DIR = Path(os.environ.get("NYAYAMIND_DATA_DIR", PROJECT_ROOT / "data"))
FEED_DIR = Path(os.environ.get("NYAYAMIND_FEED_DIR", DATA_DIR / "vespa_feed"))

CA_CSV = DATA_DIR / "CA" / "Central_Acts.csv"
SA_DIR = DATA_DIR / "SA"
SC_CSV = DATA_DIR / "SC" / "SC_data.csv"
HC_CSV = DATA_DIR / "HC" / "hc_cases_judgements.csv"
TEST_JSON = DATA_DIR / "test_data" / "test_v2.json"

VESPA_PORT = int(os.environ.get("VESPA_PORT", "8085"))
VESPA_CFG_PORT = int(os.environ.get("VESPA_CFG_PORT", "19071"))
VESPA_URL = os.environ.get("VESPA_URL", f"http://localhost:{VESPA_PORT}")
APP_NAME = "legalknowledgebase"

MODEL_NAME = os.environ.get(
    "NYAYAMIND_EMBED_MODEL", "Snowflake/snowflake-arctic-embed-m-v2.0"
)
EMBED_DIM = 768
CHUNK_TOKENS = 8192
RRF_K = 60
CANDIDATE_HITS = 10  # top-10 dense + top-10 BM25 per query (paper setting)

# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------
CA_SCHEMA = "centralacts"
SC_SCHEMA = "scjudgments"
HC_SCHEMA = "hcjudgments"

STATE_LIST = [
    "Lakshadweep", "Madhya Pradesh", "Tamil Nadu", "Chandigarh", "Bihar",
    "Uttarakhand", "Rajasthan", "Jharkhand", "Jammu and Kashmir", "Odisha",
    "Puducherry", "Punjab", "Andhra Pradesh", "West Bengal", "Tripura",
    "Ladakh", "Chhattisgarh", "Maharashtra", "Telangana", "Himachal Pradesh",
    "Gujarat", "Manipur", "Karnataka", "Arunachal Pradesh",
    "Andaman and Nicobar Islands", "Assam", "Delhi", "Goa", "Meghalaya",
    "Dadra and Nagar Haveli and Daman and Diu", "Kerala", "Haryana",
    "Uttar Pradesh",
]


def state_schema(state_name: str) -> str:
    """'Tamil Nadu' -> 'sa_tamil_nadu' (same rule as the original setup)."""
    return "sa_" + state_name.lower().replace(" ", "_")


def state_csv(state_name: str) -> Path:
    return SA_DIR / f"{state_name}.csv"


def available_states() -> list[str]:
    return [s for s in STATE_LIST if state_csv(s).exists()]


def sa_schemas() -> list[str]:
    return [state_schema(s) for s in available_states()]


def all_schemas() -> list[str]:
    return [CA_SCHEMA, SC_SCHEMA, HC_SCHEMA, *sa_schemas()]


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------
csv.field_size_limit(2**31 - 1)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def preprocess_text(text: Any, lower: bool = True) -> str:
    """Whitespace-collapse, ASCII-fold and (optionally) lowercase a text."""
    if text is None:
        return ""
    text = str(text)
    if not text or text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _CONTROL_CHARS.sub(" ", text)  # Vespa rejects control code points
    if lower:
        text = text.lower()
    return " ".join(text.split())


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    """Stream a (possibly multi-GB) CSV file one row at a time."""
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


_STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be
    because been before being below between both but by can could did do does
    doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more
    most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves said shall may upon
    whether thereof therein hereby within without per vs v""".split()
)


def bm25_query_terms(text: str, max_terms: int = 96) -> str:
    """Turn free text into a safe weakAnd query string.

    Long case facts contain quotes, brackets, '+'/'-' and other characters
    that Vespa's query parser treats as syntax.  Keep only alphanumeric
    tokens, drop stop words and duplicates, and cap the number of terms so
    that weakAnd stays fast.
    """
    seen: dict[str, None] = {}
    for token in re.findall(r"[a-z0-9]+", preprocess_text(text)):
        if len(token) < 2 or token in _STOPWORDS or token in seen:
            continue
        seen[token] = None
        if len(seen) >= max_terms:
            break
    return " ".join(seen)


# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------
_MODEL = None


def _repair_nonpersistent_buffers(model) -> None:
    """Rebuild buffers that are not stored in the checkpoint.

    Transformers >= 5 instantiates remote-code models on the meta device and
    only materialises tensors that are present in the checkpoint.  The GTE
    backbone of arctic-embed-m-v2.0 keeps ``position_ids`` and the RoPE
    ``inv_freq``/``cos_cached``/``sin_cached`` as non-persistent buffers, which
    therefore end up uninitialised (garbage indices -> CUDA device asserts, or
    silently wrong embeddings).  Recompute them exactly as the model's own
    ``__init__`` does.
    """
    import types

    import torch

    backbone = model[0].auto_model
    device = next(backbone.parameters()).device
    if not hasattr(backbone, "get_extended_attention_mask"):
        # Removed from PreTrainedModel in transformers 5; the GTE code still
        # calls it.  Same semantics as transformers 4.x for 2-D masks.
        def get_extended_attention_mask(self, attention_mask, input_shape=None, *_):
            dtype = next(self.parameters()).dtype
            mask = attention_mask[:, None, None, :].to(dtype)
            return (1.0 - mask) * torch.finfo(dtype).min

        backbone.get_extended_attention_mask = types.MethodType(
            get_extended_attention_mask, backbone
        )
    for module in backbone.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "_set_cos_sin_cache"):
            inv_freq = 1.0 / (
                module.base
                ** (torch.arange(0, module.dim, 2, device=device).float() / module.dim)
            )
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module._set_cos_sin_cache(
                seq_len=module.max_position_embeddings,
                device=device,
                dtype=torch.float32,
            )
        if hasattr(module, "position_ids") and hasattr(module, "word_embeddings"):
            max_pos = backbone.config.max_position_embeddings
            module.register_buffer(
                "position_ids", torch.arange(max_pos, device=device), persistent=False
            )


def load_embedder(device: str | None = None):
    """Load arctic-embed-m-v2.0 once (fp16 on GPU, SDPA attention).

    Set ``HF_HUB_OFFLINE=1`` after the first download to run fully offline.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    kwargs = dict(
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        # xformers is not available on Windows/py3.13; SDPA is equivalent.
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
    )
    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True, **kwargs)
    except Exception:  # not cached yet -> download once
        model = SentenceTransformer(MODEL_NAME, **kwargs)
    _repair_nonpersistent_buffers(model)
    model.max_seq_length = CHUNK_TOKENS
    model.eval()
    _MODEL = model
    print(f"Loaded {MODEL_NAME} on {device} ({dtype})", file=sys.stderr)
    return model


def chunk_by_tokens(
    text: str,
    tokenizer,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = 100,
    max_chunks: int | None = None,
) -> list[str]:
    """Split text into <= chunk_tokens windows using tokenizer offsets.

    One tokenizer pass per document (instead of the repeated re-tokenising
    done by a recursive splitter), so chunking a 9 GB corpus stays cheap.
    Window ends are nudged back to a sentence boundary when one is close.
    """
    if not text:
        return []
    budget = chunk_tokens - 2  # room for <s> and </s>
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        verbose=False,
    )
    offsets = enc["offset_mapping"]
    if len(offsets) <= budget:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        if end < len(offsets):
            # Prefer to cut after a full stop within the last 10% of the window.
            floor = start + int(budget * 0.9)
            for cut in range(end - 1, floor, -1):
                if text[offsets[cut][0] : offsets[cut][1]].endswith("."):
                    end = cut + 1
                    break
        chunks.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end >= len(offsets) or (max_chunks and len(chunks) >= max_chunks):
            break
        start = max(end - overlap, start + 1)
    return chunks


def head_by_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Return the prefix of ``text`` that fits in ``max_tokens`` model tokens.

    Only a character prefix is tokenised, so this stays cheap for long texts.
    """
    budget = max_tokens - 2
    prefix = text[: max_tokens * 8]
    offsets = tokenizer(
        prefix, add_special_tokens=False, return_offsets_mapping=True,
        return_attention_mask=False, verbose=False,
    )["offset_mapping"]
    if len(offsets) <= budget and len(prefix) == len(text):
        return text
    return prefix[: offsets[min(budget, len(offsets)) - 1][1]] if offsets else ""


def embed_texts(
    texts: Sequence[str],
    is_query: bool = False,
    token_budget: int = 16384,
    max_batch: int = 64,
):
    """Embed texts with length-sorted, token-budgeted batches.

    Returns an (n, 768) float32 numpy array of L2-normalised vectors.
    Queries use the model's ``query`` prompt, as required by arctic-embed v2.
    """
    import numpy as np
    import torch

    model = load_embedder()
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    # Cheap upper-bound length estimate (legal English is ~4 chars/token);
    # only used for sorting and batch sizing, so no second tokenizer pass.
    lengths = [min(len(t) // 3 + 2, CHUNK_TOKENS) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    position = 0
    while position < len(order):
        longest = lengths[order[position]]
        size = max(1, min(max_batch, token_budget // max(longest, 1)))
        batch_ids = order[position : position + size]
        with torch.inference_mode():
            vectors = model.encode(
                [texts[i] for i in batch_ids],
                batch_size=len(batch_ids),
                prompt_name="query" if is_query else None,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        out[batch_ids] = vectors.astype(np.float32)
        position += size
    return out


def bf16_hex(vector) -> str:
    """Encode a float vector as Vespa's hex format for tensor<bfloat16>.

    bfloat16 = upper 16 bits of an IEEE float32 (round-to-nearest-even).
    """
    import numpy as np

    as_int = np.asarray(vector, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((as_int + 0x7FFF + ((as_int >> 16) & 1)) >> 16).astype(">u2")
    return rounded.tobytes().hex()


# --------------------------------------------------------------------------
# Offline feed shards (pre-processing stage)
# --------------------------------------------------------------------------
class ShardWriter:
    """Write Vespa feed operations to resumable gzip JSONL shards.

    A shard is only renamed to its final name after it has been fully written,
    together with a progress marker recording the last source row consumed.
    Re-running an ingestion script therefore skips everything that has
    already been embedded.
    """

    def __init__(self, schema: str, shard_size: int = 2000):
        self.dir = FEED_DIR / schema
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffer: list[str] = []
        self.progress_path = self.dir / "_progress.json"
        progress = (
            json.loads(self.progress_path.read_text())
            if self.progress_path.exists()
            else {"next_row": 0, "shards": 0, "docs": 0}
        )
        self.next_row = progress["next_row"]
        self.shards = progress["shards"]
        self.docs = progress["docs"]
        for stale in self.dir.glob("*.tmp"):
            stale.unlink()

    @property
    def done(self) -> bool:
        return (self.dir / "_complete").exists()

    def add(self, doc_id: str, fields: dict[str, Any]) -> None:
        self.buffer.append(json.dumps({"id": doc_id, "fields": fields}, ensure_ascii=False))

    def commit(self, next_row: int, force: bool = False) -> None:
        """Flush the buffer as a shard once it is big enough (or forced)."""
        if not self.buffer or (len(self.buffer) < self.shard_size and not force):
            return
        final = self.dir / f"shard_{self.shards:06d}.jsonl.gz"
        tmp = final.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as handle:
            handle.write("\n".join(self.buffer) + "\n")
        os.replace(tmp, final)
        self.docs += len(self.buffer)
        self.shards += 1
        self.next_row = next_row
        self.buffer = []
        self._save()

    def close(self, next_row: int) -> None:
        self.commit(next_row, force=True)
        self.next_row = next_row
        self._save()
        (self.dir / "_complete").write_text(json.dumps({"docs": self.docs}))

    def _save(self) -> None:
        self.progress_path.write_text(
            json.dumps({"next_row": self.next_row, "shards": self.shards, "docs": self.docs})
        )


def _shard_paths(schema: str, follow: bool, poll_seconds: int = 20) -> Iterator[Path]:
    """Yield shards in order; with ``follow``, wait for new ones until the
    embedding stage marks the schema complete."""
    import time

    shard_dir = FEED_DIR / schema
    index = 0
    while True:
        path = shard_dir / f"shard_{index:06d}.jsonl.gz"
        if path.exists():
            yield path
            index += 1
            continue
        if not follow:
            return
        if (shard_dir / "_complete").exists():
            if path.exists():  # written just before _complete
                continue
            return
        time.sleep(poll_seconds)


def iter_feed_shards(schema: str, follow: bool = False) -> Iterator[dict[str, Any]]:
    for shard in _shard_paths(schema, follow):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    op = json.loads(line)
                    op["fields"] = {
                        k: _CONTROL_CHARS.sub(" ", v) if isinstance(v, str) else v
                        for k, v in op["fields"].items()
                    }
                    yield op


def run_embedding_stage(
    schema: str,
    rows: Iterable[tuple[int, dict[str, str]]],
    make_docs,
    total: int | None = None,
    limit: int | None = None,
    docs_per_batch: int = 256,
) -> None:
    """Chunk + embed a corpus into feed shards.

    ``make_docs(row_index, row, tokenizer)`` returns a list of
    ``(doc_id, fields_without_embedding, text_to_embed)`` tuples.
    """
    from tqdm import tqdm

    writer = ShardWriter(schema)
    if writer.done:
        print(f"[{schema}] embeddings already complete ({writer.docs} docs) - skipping.")
        return
    tokenizer = load_embedder().tokenizer
    start = writer.next_row
    if start:
        print(f"[{schema}] resuming after source row {start} ({writer.docs} docs done).")
    pending: list[tuple[str, dict[str, Any], str]] = []
    last_row = start

    def flush(next_row: int) -> None:
        if not pending:
            return
        vectors = embed_texts([text for _, _, text in pending])
        for (doc_id, fields, _), vector in zip(pending, vectors):
            fields["embedding"] = {"values": bf16_hex(vector)}
            writer.add(doc_id, fields)
        pending.clear()
        writer.commit(next_row)

    # CSV parsing, text cleaning and tokenisation run in a producer thread so
    # the GPU is not idle while the next batch is prepared.
    import queue
    import threading

    work: queue.Queue = queue.Queue(maxsize=8)
    error: list[BaseException] = []

    def produce() -> None:
        batch: list[tuple[str, dict[str, Any], str]] = []
        next_row = start
        try:
            for row_index, row in rows:
                if row_index < start:
                    continue
                if limit is not None and row_index >= limit:
                    break
                batch.extend(make_docs(row_index, row, tokenizer))
                next_row = row_index + 1
                if len(batch) >= docs_per_batch:
                    work.put((batch, next_row, next_row - start))
                    batch = []
            work.put((batch, next_row, next_row - start))
        except BaseException as exc:  # surface errors in the main thread
            error.append(exc)
        finally:
            work.put(None)

    threading.Thread(target=produce, daemon=True).start()
    progress = tqdm(total=total, initial=start, desc=f"embed {schema}", unit="row")
    rows_seen = 0
    while (item := work.get()) is not None:
        batch, last_row, seen = item
        pending.extend(batch)
        flush(last_row)
        progress.update(seen - rows_seen)
        rows_seen = seen
    progress.close()
    if error:
        raise error[0]
    if limit is None:
        writer.close(last_row)
        print(f"[{schema}] embedding complete: {writer.docs} docs in {writer.shards} shards.")
    else:
        writer.commit(last_row, force=True)
        print(f"[{schema}] embedded first {limit} rows ({writer.docs} docs); not marked complete.")


# --------------------------------------------------------------------------
# Vespa access
# --------------------------------------------------------------------------
def vespa_app():
    from vespa.application import Vespa

    return Vespa(url=VESPA_URL)


def run_feed_stage(
    schema: str, max_workers: int = 8, max_connections: int = 4, follow: bool = False
) -> None:
    """Send the pre-computed shards of one schema to Vespa.

    ``follow=True`` runs alongside the embed stage, feeding each shard as
    soon as it is written and stopping once the schema is marked complete.
    """
    import time

    from tqdm import tqdm

    shard_dir = FEED_DIR / schema
    while follow and not shard_dir.exists():
        time.sleep(20)
    if not shard_dir.exists():
        print(f"[{schema}] no feed shards found in {shard_dir}; run the embed stage first.")
        return
    progress_path = shard_dir / "_progress.json"
    total = None if follow else (
        json.loads(progress_path.read_text())["docs"] if progress_path.exists() else None
    )
    app = vespa_app()
    failures: list[str] = []
    bar = tqdm(total=total, desc=f"feed {schema}", unit="doc")

    def callback(response, doc_id):
        bar.update(1)
        if not response.is_successful():
            failures.append(f"{doc_id}: {response.get_json()}")

    app.feed_iterable(
        iter_feed_shards(schema, follow=follow),
        schema=schema,
        namespace=schema,
        callback=callback,
        max_queue_size=2000,
        max_workers=max_workers,
        max_connections=max_connections,
    )
    bar.close()
    if failures:
        print(f"[{schema}] {len(failures)} documents failed, e.g.: {failures[:3]}")
    else:
        print(f"[{schema}] feed complete.")


def add_stage_args(parser) -> None:
    parser.add_argument(
        "--stage",
        choices=("embed", "feed", "all"),
        default="all",
        help="embed: chunk+embed into offline feed shards (no Vespa needed); "
        "feed: push shards into Vespa; all: both (default).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N source rows (smoke tests)."
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="With --stage feed: keep feeding new shards while a separate embed run "
        "is still writing them (overlaps GPU embedding with Vespa indexing).",
    )
    parser.add_argument(
        "--embed-tokens",
        type=int,
        default=CHUNK_TOKENS,
        help="Embed only the first N tokens of each stored chunk (BM25 still indexes "
        "the whole chunk). Lower = faster on a laptop GPU. Default 8192 (paper).",
    )


# --------------------------------------------------------------------------
# Hybrid search (dense + BM25, fused by RRF)
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: Sequence[Sequence[dict]], k: int = RRF_K) -> list[dict]:
    """Fuse ranked hit lists; each fused hit gets an ``rrf_score`` key."""
    fused: dict[str, dict[str, Any]] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            entry = fused.setdefault(hit["id"], {"hit": hit, "score": 0.0})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [{**item["hit"], "rrf_score": item["score"]} for item in ranked]


def hybrid_search(
    sources: Sequence[str],
    query_text: str,
    query_embedding=None,
    hits: int = CANDIDATE_HITS,
    summary_fields: str = "*",
    app=None,
    timeout: str = "20s",
) -> dict[str, list[dict]]:
    """Run the paper's two parallel queries against ``sources``.

    Returns ``{"vector": [...], "bm25": [...], "fused": [...]}``.
    ``query_embedding`` may be supplied (pre-computed offline); otherwise the
    query text is embedded on the fly with the ``query`` prompt.
    """
    app = app or vespa_app()
    if query_embedding is None:
        query_embedding = embed_texts([query_text], is_query=True)[0]
    sources_sql = ", ".join(sources)
    vector_body = {
        "yql": (
            f"select {summary_fields} from sources {sources_sql} where "
            f"{{targetHits:{hits}}}nearestNeighbor(embedding, q_emb)"
        ),
        "ranking.profile": "semantic",
        "input.query(q_emb)": [float(x) for x in query_embedding],
        "hits": hits,
        "timeout": timeout,
    }
    terms = bm25_query_terms(query_text)
    bm25_body = {
        "yql": f"select {summary_fields} from sources {sources_sql} where userQuery()",
        "query": terms,
        "model.type": "weakAnd",
        "ranking.profile": "bm25",
        "hits": hits,
        "timeout": timeout,
    }
    vector_hits = _query_hits(app, vector_body)
    bm25_hits = _query_hits(app, bm25_body) if terms else []
    return {
        "vector": vector_hits,
        "bm25": bm25_hits,
        "fused": reciprocal_rank_fusion([vector_hits, bm25_hits]),
    }


def search_collection(
    sources: Sequence[str],
    query_text: str,
    top_k: int = 5,
    query_embedding=None,
    dedupe_fields: Sequence[str] | None = None,
    exclude=None,
    hits: int = CANDIDATE_HITS,
    app=None,
) -> list[dict[str, Any]]:
    """Hybrid search returning flat, ranked result records.

    ``dedupe_fields``: keep only the best chunk per value of these fields
    (e.g. one result per judgment).  ``exclude(fields) -> bool`` drops hits,
    e.g. the query case's own judgment.
    """
    result = hybrid_search(sources, query_text, query_embedding, hits=hits, app=app)
    vector_rank = {hit["id"]: rank for rank, hit in enumerate(result["vector"], 1)}
    bm25_rank = {hit["id"]: rank for rank, hit in enumerate(result["bm25"], 1)}
    records: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for hit in result["fused"]:
        fields = hit.get("fields", {})
        if exclude is not None and exclude(fields):
            continue
        if dedupe_fields:
            key = tuple(fields.get(name) for name in dedupe_fields)
            if key in seen:
                continue
            seen.add(key)
        records.append(
            {
                "rank": len(records) + 1,
                "schema": schema_of_hit(hit),
                "vespa_id": hit["id"],
                "rrf_score": round(hit["rrf_score"], 6),
                "vector_rank": vector_rank.get(hit["id"]),
                "bm25_rank": bm25_rank.get(hit["id"]),
                **{k: v for k, v in fields.items() if k not in ("sddocname", "documentid", "summaryfeatures")},
            }
        )
        if len(records) >= top_k:
            break
    return records


def statute_title(record: dict[str, Any]) -> str:
    """IndianKanoon-style title, e.g. 'Section 80IA in The Income Tax Act, 1961'."""
    number = str(record.get("section_number", "")).strip()
    number = re.sub(r"^(\d+)\.0$", r"\1", number).rstrip(".")  # CSV float artefacts: '2.0' -> '2'
    statute = str(record.get("name_of_statute", "")).strip()
    return f"Section {number} in {statute}" if number else statute


def _query_hits(app, body: dict[str, Any]) -> list[dict]:
    response = app.query(body=body)
    if not response.is_successful():
        raise RuntimeError(f"Vespa query failed: {json.dumps(response.json)[:500]}")
    return list(response.hits or [])


def schema_of_hit(hit: dict) -> str:
    """Vespa hit ids look like 'id:<namespace>:<schema>::<local id>'."""
    parts = str(hit.get("id", "")).split(":")
    return parts[2] if len(parts) > 2 else ""
