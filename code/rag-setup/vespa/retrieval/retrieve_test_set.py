"""Offline NyayaMind retrieval for the test set (Vespa hybrid, v2).

Input (as in the NyayaMind paper, "At inference time, NyayaMind takes as input
the factual description of a legal case"): only the case's
``Simplified_Facts`` (``Facts`` as fallback).  Issues, arguments, reasoning,
the decision and the gold citations are never used to retrieve.  The
judgment's decision date is used only as a temporal cut-off (a judgment cannot
cite a precedent decided after it, nor an Act enacted after it).

v1 ran one paper-style hybrid query (top-10 dense + top-10 BM25, RRF k=60)
over the whole facts text and kept the fused top-5.  v2 keeps that query and
adds evidence lists that are fused with weighted RRF:

statutes (Central Acts + State Acts)
    explicit      "Section 3 of the 1971 Act", "Sections 279, 337 IPC" ...
                  in the facts, resolved against the corpus (statute_refs.py)
    facts         dense + BM25 over the whole facts text (the v1 query)
    passages      dense + BM25 per facts passage (long facts dilute one vector)
    named_act     dense + BM25 restricted to the Acts named in the facts
    propagated    sections cited in the most similar SC/HC judgments
precedents (Supreme Court + High Courts)
    facts         dense + BM25 over the whole facts text
    passages      dense + BM25 per facts passage
    propagated    cases cited ("X v. Y") in the most similar judgments,
                  resolved against the corpus (case_refs.py)

The case's own judgment (and lower-court judgments in the same matter) are
removed by party-name / verbatim-text match, before propagation as well.

Stages (offline pre-processing; nothing runs during model inference):
    --stage queries   split facts into passages, embed (GPU) -> <out>/queries.json, query_embeddings.npy
    --stage retrieve  query Vespa                            -> <out>/candidates.jsonl (resumable)
    --stage merge     fuse, cut off, write outputs + report  -> <out>/test_v2_rag.json
                                                                 <out>/test_v2_rag_inference.json
                                                                 <out>/retrieval_report.json
    --stage all       all three (default)

The inference file has the test_v2.json layout (retrieved items in
section_titles/section_texts/cited_cases/cited_case_judgments), e.g.
    python code/inference/inference_Claude_Opus_5_5.py --dataset data/test_data/retrieval_vespa_v2/test_v2_rag_inference.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_refs as cr  # noqa: E402
import common as c  # noqa: E402
import statute_refs as sr  # noqa: E402

QUERY_FIELDS = ("Simplified_Facts", "Facts")
STRATEGY = "vespa-hybrid-rrf-v2"

# Weighted RRF over the evidence lists (hand-set defaults; --weights overrides).
DEFAULT_WEIGHTS = {
    "statutes": {
        "explicit": 4.0,
        "facts_dense": 1.0,
        "facts_bm25": 0.5,
        "passage_dense": 1.0,
        "passage_bm25": 0.5,
        "named_dense": 1.0,
        "named_bm25": 1.0,
        "propagated": 2.0,
    },
    "precedents": {
        "facts_dense": 1.0,
        "facts_bm25": 1.0,
        "passage_dense": 0.75,
        "passage_bm25": 0.5,
        "propagated": 2.0,
    },
}


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def iter_json_array(path: Path, chunk_size: int = 4 << 20) -> Iterator[dict[str, Any]]:
    """Stream the records of a large JSON array file."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as handle:
        buffer = handle.read(chunk_size)
        position = buffer.index("[") + 1
        eof = False
        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer) or eof:
                    break
                more = handle.read(chunk_size)
                eof = not more
                buffer, position = buffer[position:] + more, 0
            if position >= len(buffer) or buffer[position] == "]":
                return
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = handle.read(chunk_size)
                if not more:
                    raise
                buffer, position = buffer[position:] + more, 0
                continue
            yield record
            position = end
            if position > chunk_size:
                buffer, position = buffer[position:], 0


def iter_cases(args) -> Iterator[tuple[int, dict[str, Any]]]:
    for index, record in enumerate(iter_json_array(args.dataset)):
        if index < args.start:
            continue
        if args.limit is not None and index >= args.start + args.limit:
            return
        yield index, record


def query_text(record: dict[str, Any]) -> tuple[str, str]:
    for name in QUERY_FIELDS:
        value = str(record.get(name) or "").strip()
        if value:
            return name, value
    return "", ""


def split_passages(text: str, target_chars: int = 700, max_passages: int = 8) -> list[str]:
    """Sentence-aligned windows of ~target_chars over the facts."""
    text = re.sub(r"(?m)^\s*Matter:\s*", "", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.;:?!])\s+|\n+", text) if len(s.strip()) > 2]
    passages, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > target_chars:
            passages.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        passages.append(current)
    if len(passages) <= 1:
        return []  # short facts: the whole-facts query already covers it
    if len(passages) > max_passages:  # merge neighbours evenly
        step = len(passages) / max_passages
        passages = [" ".join(passages[int(i * step): int((i + 1) * step)]) for i in range(max_passages)]
    return passages


def decision_date(record: dict[str, Any]) -> tuple[str, str]:
    """(dd-mm-yyyy, source) of the judgment being predicted; used only as a cut-off."""
    head = str(record.get("full_text") or "")[:600]
    m = re.search(r"\bon (\d{1,2} [A-Za-z]+,? \d{4})", head)
    if m and cr.norm_date(m.group(1)):
        return cr.norm_date(m.group(1)), "full_text"
    m = re.match(r"(\d{4})_", str(record.get("document_id") or ""))
    if m:
        return f"31-12-{m.group(1)}", "document_id_year"
    return "", "unknown"


# --------------------------------------------------------------------------
# Stage 1: queries
# --------------------------------------------------------------------------
def stage_queries(args) -> None:
    import numpy as np

    cases, texts = [], []
    for index, record in iter_cases(args):
        field, text = query_text(record)
        passages = split_passages(text) if text else []
        date, date_source = decision_date(record)
        cases.append(
            {
                "index": index,
                "key": str(record.get("document_id") or "").strip() or f"idx_{index}",
                "case_name": record.get("Case Name", ""),
                "query_field": field,
                "decision_date": date,
                "decision_date_source": date_source,
                "text": text,
                "passages": passages,
                "vector_offset": len(texts),
            }
        )
        texts.extend([text or " ", *passages])
    print(f"Embedding {len(cases)} facts queries + {len(texts) - len(cases)} passages ...", flush=True)
    started = time.time()
    vectors = c.embed_texts(texts, is_query=True)
    np.save(args.out_dir / "query_embeddings.npy", vectors)
    (args.out_dir / "queries.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    print(f"Query embeddings saved ({time.time() - started:.0f}s).")


# --------------------------------------------------------------------------
# Stage 2: retrieve candidate lists
# --------------------------------------------------------------------------
def local_id(hit: dict) -> str:
    return str(hit.get("id", "")).split("::", 1)[-1]


def hybrid(app, sources, text, vector, hits, where="", bm25=True):
    """Paper-style parallel dense + BM25 queries; returns (dense_hits, bm25_hits)."""
    sources_sql = ", ".join(sources)
    extra = f" and ({where})" if where else ""
    dense = c._query_hits(app, {
        "yql": f"select * from sources {sources_sql} where "
               f"({{targetHits:{hits}}}nearestNeighbor(embedding, q_emb)){extra}",
        "ranking.profile": "semantic",
        "input.query(q_emb)": [float(x) for x in vector],
        "hits": hits,
        "timeout": "30s",
    })
    terms = c.bm25_query_terms(text) if bm25 else ""
    lexical = c._query_hits(app, {
        "yql": f"select * from sources {sources_sql} where userQuery(){extra}",
        "query": terms,
        "model.type": "weakAnd",
        "ranking.profile": "bm25",
        "hits": hits,
        "timeout": "30s",
    }) if terms else []
    return dense, lexical


def act_filter(catalog: sr.Catalog, act: tuple[str, ...]) -> str:
    name = catalog.entries[catalog.acts[act][0]]["name_of_statute"]
    tokens = [t for t in re.findall(r"[a-z0-9]+", c.preprocess_text(name)) if t != "the"][:12]
    return "name_of_statute contains phrase(" + ", ".join(json.dumps(t) for t in tokens) + ")" if tokens else ""


class Retriever:
    def __init__(self, args):
        import numpy as np

        self.args = args
        self.app = c.vespa_app()
        self.vectors = np.load(args.out_dir / "query_embeddings.npy")
        print("Loading statute catalogue ...", flush=True)
        self.statutes = sr.Catalog.load()
        self.statute_index = {e["id"]: i for i, e in enumerate(self.statutes.entries)}
        print("Loading case catalogue ...", flush=True)
        self.cases = cr.CaseCatalog.load()
        self.case_by_id, self.case_by_name = {}, {}
        for i, e in enumerate(self.cases.entries):
            self.case_by_id[e["id"]] = i
            self.case_by_name.setdefault((e["case_name"], e["judgment_date"]), i)
        self.statute_sources = [c.CA_SCHEMA, *c.sa_schemas()]
        self.case_sources = [c.SC_SCHEMA, c.HC_SCHEMA]
        self.resolve_cache: dict[tuple, int | None] = {}

    # -- id mapping --------------------------------------------------------
    def statute_ids(self, hits) -> list[int]:
        out = []
        for hit in hits:
            i = self.statute_index.get(local_id(hit))
            if i is not None and i not in out:
                out.append(i)
        return out

    def case_idx(self, hit) -> int | None:
        f = hit.get("fields", {})
        i = self.case_by_id.get(local_id(hit))
        if i is None:
            i = self.case_by_name.get((f.get("case_name", ""), cr.norm_date(f.get("judgment_date", ""))))
        return i

    # -- exclusion ----------------------------------------------------------
    def case_filter(self, record, case_meta):
        own = record.get("Case Name", "")
        own_left, own_right = cr.split_parties(own)
        cutoff = cr.date_ord(case_meta["decision_date"])
        facts = c.preprocess_text(record.get("Facts") or record.get("Text") or "")
        probes = [facts[i: i + 120] for i in (len(facts) // 4, len(facts) // 2) if len(facts) > 400]

        def excluded(name: str, date: str, body: str = "") -> bool:
            if cutoff and date and cr.date_ord(cr.norm_date(date)) >= cutoff:
                return True  # decided on/after the judgment being predicted
            left, right = cr.split_parties(name)
            if own_left and right and cr.jaccard(own_left, left) >= 0.5 and cr.jaccard(own_right, right) >= 0.5:
                return True  # same parties: own judgment / same matter below
            if own_left and right and cr.jaccard(own_left, right) >= 0.5 and cr.jaccard(own_right, left) >= 0.5:
                return True
            if body and probes:
                text = c.preprocess_text(body)
                return any(p in text for p in probes)
            return False

        return excluded

    # -- one case -----------------------------------------------------------
    def run(self, case_meta: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        args, app = self.args, self.app
        text = case_meta["text"]
        off = case_meta["vector_offset"]
        facts_vec = self.vectors[off]
        passage_vecs = self.vectors[off + 1: off + 1 + len(case_meta["passages"])]
        excluded = self.case_filter(record, case_meta)
        cutoff = cr.date_ord(case_meta["decision_date"])

        st: dict[str, Any] = {}
        pr: dict[str, Any] = {}

        # statutes: explicit references
        refs, named_acts = sr.extract_references(text, self.statutes)
        st["explicit"] = sr.resolve_sections(refs, self.statutes)

        # statutes: whole facts / passages / named acts
        d, b = hybrid(app, self.statute_sources, text, facts_vec, args.hits)
        st["facts_dense"], st["facts_bm25"] = self.statute_ids(d), self.statute_ids(b)
        st["passage_dense"], st["passage_bm25"] = [], []
        for passage, vec in zip(case_meta["passages"], passage_vecs):
            d, b = hybrid(app, self.statute_sources, passage, vec, args.passage_hits)
            st["passage_dense"].append(self.statute_ids(d))
            st["passage_bm25"].append(self.statute_ids(b))
        st["named_dense"], st["named_bm25"] = [], []
        for act in named_acts[:4]:
            where = act_filter(self.statutes, act)
            if not where:
                continue
            d, b = hybrid(app, self.statute_sources, text, facts_vec, args.passage_hits, where=where)
            st["named_dense"].append(self.statute_ids(d))
            st["named_bm25"].append(self.statute_ids(b))

        # precedents: whole facts / passages
        best_chunk: dict[int, str] = {}
        bodies: dict[int, str] = {}

        def case_ids(hits) -> list[int]:
            out = []
            for hit in hits:
                f = hit.get("fields", {})
                i = self.case_idx(hit)
                if i is None or i in out:
                    continue
                body = f.get("content") or f.get("text") or ""
                if excluded(f.get("case_name", ""), f.get("judgment_date", ""), body):
                    continue
                out.append(i)
                best_chunk.setdefault(i, local_id(hit))
                if len(body) > len(bodies.get(i, "")):
                    bodies[i] = body
            return out

        d, b = hybrid(app, self.case_sources, text, facts_vec, args.case_hits)
        pr["facts_dense"], pr["facts_bm25"] = case_ids(d), case_ids(b)
        pr["passage_dense"], pr["passage_bm25"] = [], []
        for passage, vec in zip(case_meta["passages"], passage_vecs):
            d, b = hybrid(app, self.case_sources, passage, vec, args.passage_hits)
            pr["passage_dense"].append(case_ids(d))
            pr["passage_bm25"].append(case_ids(b))

        # propagation: what do the most similar judgments cite?
        similar = [
            item["id"] for item in c.reciprocal_rank_fusion(
                [[{"id": i} for i in pr["facts_dense"]], [{"id": i} for i in pr["facts_bm25"]]]
            )
        ][: args.propagate_from]
        st_votes: dict[int, float] = defaultdict(float)
        pr_votes: dict[int, float] = defaultdict(float)
        for rank, source in enumerate(similar):
            body = bodies.get(source, "")
            weight = 1.0 / (10 + rank)
            refs_s, _ = sr.extract_references(body, self.statutes)
            for i in sr.resolve_sections(refs_s, self.statutes):
                st_votes[i] += weight
            for left, right, year in cr.extract_case_mentions(body):
                key = (frozenset(left), frozenset(right), year, cutoff)
                if key not in self.resolve_cache:
                    self.resolve_cache[key] = self.cases.resolve(left, right, year, before=cutoff)
                j = self.resolve_cache[key]
                if j is None or j == source:
                    continue
                e = self.cases.entries[j]
                if excluded(e["case_name"], e["judgment_date"]):
                    continue
                pr_votes[j] += weight
                best_chunk.setdefault(j, e["id"])
        st["propagated"] = [[i, round(v, 5)] for i, v in sorted(st_votes.items(), key=lambda x: -x[1])][:50]
        pr["propagated"] = [[i, round(v, 5)] for i, v in sorted(pr_votes.items(), key=lambda x: -x[1])][:50]

        return {
            **{k: case_meta[k] for k in ("index", "key", "case_name", "query_field", "decision_date",
                                         "decision_date_source")},
            "n_passages": len(case_meta["passages"]),
            "explicit_refs": [{"kind": r.kind, "number": r.number, "act": r.act_mention} for r in refs][:40],
            "named_acts": [" ".join(a) for a in named_acts[:8]],
            "propagated_from": similar,
            "statutes": st,
            "precedents": pr,
            "best_chunk": {str(k): v for k, v in best_chunk.items()},
        }


def stage_retrieve(args) -> None:
    cases = json.loads((args.out_dir / "queries.json").read_text(encoding="utf-8"))
    out_path = args.out_dir / "candidates.jsonl"
    done: set[int] = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["index"])
    elif out_path.exists():
        out_path.unlink()
    by_index = {case["index"]: case for case in cases}
    todo = [i for i in by_index if i not in done and by_index[i]["text"]]
    print(f"{len(done)} cases already retrieved; {len(todo)} to go.", flush=True)
    if not todo:
        return
    retriever = Retriever(args)

    def jobs():
        wanted = set(todo)
        for index, record in enumerate(iter_json_array(args.dataset)):
            if index > max(wanted):
                return
            if index in wanted:
                yield by_index[index], record

    started, completed = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as sink, ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(lambda job: retriever.run(*job), jobs()):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            completed += 1
            if completed % 25 == 0:
                rate = completed / (time.time() - started)
                print(f"  retrieved {completed}/{len(todo)} cases ({rate:.2f}/s)", flush=True)
    print(f"Retrieval complete -> {out_path}")


# --------------------------------------------------------------------------
# Stage 3: fuse, evaluate, write
# --------------------------------------------------------------------------
def weighted_rrf(lists: dict[str, Any], weights: dict[str, float], k: int = c.RRF_K):
    """lists: name -> ranked ids | list of ranked id lists | [[id, score], ...] (propagated)."""
    scores: dict[Any, float] = defaultdict(float)
    sources: dict[Any, list[str]] = defaultdict(list)

    def add(name, ranked, weight):
        for rank, item in enumerate(ranked):
            scores[item] += weight / (k + rank + 1)
            if name not in sources[item]:
                sources[item].append(name)

    for name, value in lists.items():
        weight = weights.get(name, 0.0)
        if not weight or not value:
            continue
        if name == "propagated":
            add(name, [item for item, _ in value], weight)
        elif isinstance(value[0], list):  # one list per passage / named act -> fuse first
            inner: dict[Any, float] = defaultdict(float)
            for ranked in value:
                for rank, item in enumerate(ranked):
                    inner[item] += 1.0 / (k + rank + 1)
            add(name, sorted(inner, key=lambda x: -inner[x]), weight)
        else:
            add(name, value, weight)
    ranked = sorted(scores, key=lambda x: -scores[x])
    return ranked, scores, sources


def statute_key(catalog: sr.Catalog, i: int) -> str:
    e = catalog.entries[i]
    return f"{e['_num']}|{' '.join(e['_act'])}"


def central_acts(catalog: sr.Catalog) -> set[tuple[str, ...]]:
    """Acts in the Central Acts collection, or copied into >= 3 state collections (IPC, CrPC, ...)."""
    schemas: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for e in catalog.entries:
        schemas[e["_act"]].add(e["schema"])
    return {a for a, s in schemas.items() if c.CA_SCHEMA in s or len(s) >= 3 or a[:2] == ("constitution", "india")}


def canonical_entries(catalog: sr.Catalog) -> dict[str, int]:
    """Section key -> preferred catalogue entry (Central Acts copy, first chunk)."""
    best: dict[str, int] = {}
    for i, e in enumerate(catalog.entries):
        key = statute_key(catalog, i)
        j = best.get(key)
        if j is None or (e["schema"] == c.CA_SCHEMA and catalog.entries[j]["schema"] != c.CA_SCHEMA):
            best[key] = i
    return best


def jurisdiction(entry: dict[str, Any], central: set) -> str:
    if entry["_act"] in central:
        return "Central"
    return entry["schema"].removeprefix("sa_").replace("_", " ").title()


def statute_title(entry: dict[str, Any]) -> str:
    title = c.statute_title(entry)
    if entry["_act"][:2] == ("constitution", "india"):
        title = re.sub(r"^Section ", "Article ", title)
    return title


def pretty_date(date: str) -> str:
    """'16-01-2007' -> '16 January, 2007' (IndianKanoon style, as in the gold titles)."""
    m = re.match(r"(\d{2})-(\d{2})-(\d{4})$", date or "")
    if not m or not 1 <= int(m.group(2)) <= 12:
        return date
    month = list(cr._MONTHS)[int(m.group(2)) - 1].title()
    return f"{int(m.group(1))} {month}, {m.group(3)}"


def fuse_statutes(row, catalog: sr.Catalog, weights, top_k, canonical=None):
    ranked, scores, sources = weighted_rrf(row["statutes"], weights)
    cutoff_year = int(row["decision_date"][-4:]) if row["decision_date"] else 0
    out, seen = [], set()
    for i in ranked:
        e = catalog.entries[i]
        key = statute_key(catalog, i)
        if key in seen:  # same section from CA and SA, or another chunk
            continue
        year = sr.act_year(e["_act"])
        if cutoff_year and year and int(year) > cutoff_year:
            continue  # Act enacted after the judgment
        seen.add(key)
        out.append((canonical.get(key, i) if canonical else i, scores[i], sources[i]))
        if len(out) >= top_k:
            break
    return out


def fuse_precedents(row, weights, top_k):
    lists = {k: v for k, v in row["precedents"].items()}
    ranked, scores, sources = weighted_rrf(lists, weights)
    return [(i, scores[i], sources[i]) for i in ranked[:top_k]]


def fetch_texts(app, ids: list[str], cap: int) -> dict[str, str]:
    """Judgment text for the final precedents (Vespa document API)."""
    out = {}

    def get(local):
        schema = c.SC_SCHEMA if local.startswith("sc_") else c.HC_SCHEMA
        try:
            resp = app.get_data(schema=schema, data_id=local, namespace=schema)
            f = resp.json.get("fields", {}) if resp.is_successful() else {}
        except Exception:  # noqa: BLE001
            f = {}
        return local, (f.get("content") or f.get("text") or "")[:cap]

    with ThreadPoolExecutor(8) as pool:
        for local, text in pool.map(get, ids):
            out[local] = text
    return out


def stage_merge(args) -> None:
    weights = json.loads(args.weights.read_text()) if args.weights else DEFAULT_WEIGHTS
    rows = {}
    for line in (args.out_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if args.start <= row["index"] and (args.limit is None or row["index"] < args.start + args.limit):
                rows[row["index"]] = row
    print(f"Merging {len(rows)} cases ...", flush=True)
    statutes = sr.Catalog.load()
    cases = cr.CaseCatalog.load()
    app = c.vespa_app()

    ks = sorted({5, 10, args.top_k})
    stats = {
        "statutes": {"cases": 0, "gold": 0, "in_corpus": 0, **{f"hits@{k}": 0 for k in ks}},
        "acts": {"cases": 0, "gold": 0, "in_corpus": 0, **{f"hits@{k}": 0 for k in ks}},
        "precedents": {"cases": 0, "gold": 0, "in_corpus": 0, **{f"hits@{k}": 0 for k in ks}},
    }
    signal_hits = {"statutes": defaultdict(int), "precedents": defaultdict(int)}
    corpus_acts = {" ".join(a) for a in statutes.acts}
    central = central_acts(statutes)
    canonical = canonical_entries(statutes)

    structured, flat = [], []
    for index, record in iter_cases(args):
        row = rows.get(index)
        if row is None:
            continue
        top_st = fuse_statutes(row, statutes, weights["statutes"], args.top_k, canonical)
        top_pr = fuse_precedents(row, weights["precedents"], args.top_k)
        chunk_ids = [row["best_chunk"].get(str(i), cases.entries[i]["id"]) for i, _, _ in top_pr]
        texts = fetch_texts(app, chunk_ids, args.max_text_chars)

        st_items = []
        for rank, (i, score, src) in enumerate(top_st, 1):
            e = statutes.entries[i]
            st_items.append({
                "rank": rank,
                "title": statute_title(e),
                "act": e["name_of_statute"],
                "section_number": re.sub(r"^(\d+)\.0$", r"\1", str(e["section_number"])).rstrip("."),
                "section_title": e["section_title"],
                "jurisdiction": jurisdiction(e, central),
                "score": round(score, 6),
                "evidence": src,
                "vespa_id": e["id"],
                "text": e["section_text"][: args.max_text_chars],
            })
        pr_items = []
        for rank, ((i, score, src), chunk) in enumerate(zip(top_pr, chunk_ids), 1):
            e = cases.entries[i]
            pr_items.append({
                "rank": rank,
                "title": f"{e['case_name']} on {pretty_date(e['judgment_date'])}".strip(),
                "case_name": e["case_name"],
                "court": e["court"],
                "judgment_date": e["judgment_date"],
                "citation": e["citation"],
                "score": round(score, 6),
                "evidence": src,
                "vespa_id": chunk,
                "text": texts.get(chunk, ""),
            })

        # ---- evaluation against the gold citations ------------------------
        gold_st = list(record.get("section_titles") or [])
        gold_pr = list(record.get("cited_cases") or [])
        retrieved_keys = [statute_key(statutes, i) for i, _, _ in top_st]
        eval_st = []
        for g in gold_st:
            key = sr.norm_title_key(g)
            rank = retrieved_keys.index(key) + 1 if key in retrieved_keys else None
            eval_st.append({"gold": g, "in_corpus": key in statutes.keys, "retrieved_rank": rank})
        gold_acts = sorted({sr.norm_title_key(g).split("|", 1)[1] for g in gold_st})
        retrieved_acts = [" ".join(statutes.entries[i]["_act"]) for i, _, _ in top_st]
        eval_acts = [{"act": a, "in_corpus": a in corpus_acts,
                      "retrieved_rank": retrieved_acts.index(a) + 1 if a in retrieved_acts else None}
                     for a in gold_acts]
        eval_pr = []
        for g in gold_pr:
            rank = next((n for n, (i, _, _) in enumerate(top_pr, 1)
                         if cr.same_case(cases.entries[i]["case_name"], cases.entries[i]["judgment_date"], g)), None)
            eval_pr.append({"gold": g, "in_corpus": bool(cases.find_gold(g)), "retrieved_rank": rank})

        for name, items in (("statutes", eval_st), ("acts", eval_acts), ("precedents", eval_pr)):
            if not items:
                continue
            s = stats[name]
            s["cases"] += 1
            s["gold"] += len(items)
            s["in_corpus"] += sum(x["in_corpus"] for x in items)
            for k in ks:
                s[f"hits@{k}"] += sum(1 for x in items if x["retrieved_rank"] and x["retrieved_rank"] <= k)
        # which evidence lists contain gold anywhere (diagnostics)
        gold_keys = {sr.norm_title_key(g) for g in gold_st}
        for name, value in row["statutes"].items():
            ids = [x[0] for x in value] if name == "propagated" else (
                [i for lst in value for i in lst] if value and isinstance(value[0], list) else value)
            signal_hits["statutes"][name] += len(gold_keys & {statute_key(statutes, i) for i in ids})
        for name, value in row["precedents"].items():
            ids = [x[0] for x in value] if name == "propagated" else (
                [i for lst in value for i in lst] if value and isinstance(value[0], list) else value)
            ids = set(ids)
            signal_hits["precedents"][name] += sum(
                any(cr.same_case(cases.entries[i]["case_name"], cases.entries[i]["judgment_date"], g) for i in ids)
                for g in gold_pr)

        def recall(items, k):
            if not items:
                return None
            return round(sum(1 for x in items if x["retrieved_rank"] and x["retrieved_rank"] <= k) / len(items), 4)

        structured.append({
            "case": {
                "index": index,
                "case_number": index + 1,
                "document_id": row["key"],
                "case_name": record.get("Case Name", ""),
                "decision_date": row["decision_date"],
            },
            "query": {
                "field": row["query_field"],
                "text": query_text(record)[1],
                "explicit_references": row["explicit_refs"],
                "acts_named": row["named_acts"],
            },
            "retrieved": {"statutes": st_items, "precedents": pr_items},
            "gold": {"statutes": gold_st, "precedents": gold_pr},
            "evaluation": {
                "statutes": {f"recall@{k}": recall(eval_st, k) for k in ks} | {"items": eval_st},
                "acts": {f"recall@{k}": recall(eval_acts, k) for k in ks} | {"items": eval_acts},
                "precedents": {f"recall@{k}": recall(eval_pr, k) for k in ks} | {"items": eval_pr},
            },
        })
        legacy = {k: v for k, v in record.items()
                  if k not in ("section_titles", "section_texts", "cited_cases", "cited_case_ids",
                               "cited_case_judgments")}
        flat.append(legacy | {
            "section_titles": [x["title"] for x in st_items],
            "section_texts": [x["text"] for x in st_items],
            "cited_cases": [x["title"] for x in pr_items],
            "cited_case_ids": [x["vespa_id"] for x in pr_items],
            "cited_case_judgments": [x["text"] for x in pr_items],
            "gold_section_titles": gold_st,
            "gold_cited_cases": gold_pr,
            "retrieval_strategy": STRATEGY,
        })
        if len(structured) % 50 == 0:
            print(f"  merged {len(structured)}", flush=True)

    def summary_block(s):
        out = {"cases_with_gold": s["cases"], "gold_items": s["gold"], "gold_in_corpus": s["in_corpus"],
               "corpus_ceiling": round(s["in_corpus"] / s["gold"], 4) if s["gold"] else None}
        for k in ks:
            hits = s[f"hits@{k}"]
            out[f"hits@{k}"] = hits
            out[f"recall@{k}"] = round(hits / s["gold"], 4) if s["gold"] else None
            out[f"recall@{k}_of_findable"] = round(hits / s["in_corpus"], 4) if s["in_corpus"] else None
        return out

    report = {
        "strategy": STRATEGY,
        "cases": len(structured),
        "case_range": [args.start + 1, args.start + len(structured)] if structured else None,
        "top_k": args.top_k,
        "weights": weights,
        "metrics": {name: summary_block(s) for name, s in stats.items()},
        "gold_found_in_any_candidate_list": {k: dict(v) for k, v in signal_hits.items()},
        "notes": {
            "statutes": "gold 'Section N in Act' matched after normalising section number and Act name",
            "acts": "gold Act among the Acts of the retrieved sections",
            "precedents": "gold cited case among retrieved SC/HC judgments (party names, relaxed when dates agree)",
            "corpus_ceiling": "share of gold items that exist in the corpus at all",
            "recall@k_of_findable": "hits / gold items that exist in the corpus",
        },
    }
    meta = {"strategy": STRATEGY, "query_input": "Simplified_Facts (Facts fallback)", "top_k": args.top_k,
            "report": "retrieval_report.json"}
    with open(args.out_dir / "test_v2_rag.json", "w", encoding="utf-8") as sink:
        json.dump({"meta": meta, "cases": structured}, sink, ensure_ascii=False, indent=1)
    with open(args.out_dir / "test_v2_rag_inference.json", "w", encoding="utf-8") as sink:
        json.dump(flat, sink, ensure_ascii=False)
    (args.out_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["gold_found_in_any_candidate_list"], indent=2))
    print(f"Wrote {args.out_dir / 'test_v2_rag.json'} and test_v2_rag_inference.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=c.TEST_JSON)
    parser.add_argument("--out-dir", type=Path, default=c.DATA_DIR / "test_data" / "retrieval_vespa_v2")
    parser.add_argument("--stage", choices=("queries", "retrieve", "merge", "all"), default="all")
    parser.add_argument("--start", type=int, default=0, help="Index of the first test case (0-based).")
    parser.add_argument("--limit", type=int, default=300, help="Number of test cases (default 300: cases 1-300).")
    parser.add_argument("--top-k", type=int, default=10, help="Statutes and precedents kept per case.")
    parser.add_argument("--hits", type=int, default=30, help="Candidates per statute query (facts).")
    parser.add_argument("--case-hits", type=int, default=30, help="Candidates per precedent query (facts).")
    parser.add_argument("--passage-hits", type=int, default=10, help="Candidates per passage / named-Act query.")
    parser.add_argument("--propagate-from", type=int, default=15, help="Similar judgments mined for citations.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent cases.")
    parser.add_argument("--max-text-chars", type=int, default=8000, help="Cap on section / judgment text.")
    parser.add_argument("--weights", type=Path, default=None, help="JSON file overriding DEFAULT_WEIGHTS.")
    parser.add_argument("--overwrite", action="store_true", help="Redo retrieval for already-done cases.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("queries", "all"):
        stage_queries(args)
    if args.stage in ("retrieve", "all"):
        stage_retrieve(args)
    if args.stage in ("merge", "all"):
        stage_merge(args)


if __name__ == "__main__":
    main()
