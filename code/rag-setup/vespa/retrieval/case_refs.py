"""Precedent mentions ("X v. Y") in judgment text, resolved against the corpus.

Used for citation propagation: judgments that are similar to the query case
tend to cite the same precedents, so the case names cited inside the
retrieved judgments are resolved to Supreme Court / High Court judgments in
the corpus and voted on.

The catalogue of all corpus judgments (parties, date, court) is built once
from the offline feed shards and cached at
``data/vespa_feed/_case_catalog.json.gz``.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

CATALOG_PATH = c.FEED_DIR / "_case_catalog.json.gz"

_PARTY_NOISE = frozenset(
    """the of and m s ms mrs mr dr smt shri sri ltd limited pvt private co corp
    corporation company anr ors others another etc through its by
    represented rep no s/o d/o w/o on vs v versus appellant appellants respondent
    respondents petitioner petitioners in re thru lrs lr legal heirs deceased
    and/or since dead lr's""".split()
)
# Words too generic to identify a party on their own.
_GENERIC = frozenset(
    """state union india government govt commissioner commr commnr comm cit tax income
    central excise customs officer assistant deputy collector district authority
    board municipal corporation bank national public prosecutor police
    inspector superintendent director general secretary chief ministry
    department dept ito acit dcit addl additional sales trade""".split()
)
_ABBREV = {
    "cit": "commissioner income tax",
    "commr": "commissioner",
    "commnr": "commissioner",
    "comm": "commissioner",
    "uoi": "union india",
    "govt": "government",
    "ito": "income tax officer",
}
_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}


def party_tokens(text: str) -> set[str]:
    text = c.preprocess_text(text)
    text = re.sub(r"\bm/s\.?", " ", text)
    tokens = []
    for t in re.findall(r"[a-z]{2,}", text):
        tokens.extend(_ABBREV.get(t, t).split())
    return {t for t in tokens if t not in _PARTY_NOISE}


def split_parties(name: str) -> tuple[set[str], set[str]]:
    name = re.sub(r"\bon \d{1,2} \w+,? \d{4}\s*$", "", str(name or ""), flags=re.I)
    halves = re.split(r"\s+(?:vs?\.?|versus|v/s\.?)\s+", name, maxsplit=1, flags=re.I)
    left = party_tokens(halves[0])
    right = party_tokens(halves[1]) if len(halves) > 1 else set()
    return left, right


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def norm_date(value: str) -> str:
    """'05-08-2024' / '5/8/2024' / '5 August, 2024' -> '05-08-2024'."""
    value = str(value or "")
    m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", value)
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{m.group(3)}"
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", value)
    if m and m.group(2).lower() in _MONTHS:
        return f"{int(m.group(1)):02d}-{_MONTHS[m.group(2).lower()]:02d}-{m.group(3)}"
    return ""


def date_ord(date: str) -> int:
    """'05-08-2024' -> 20240805 (0 if unknown)."""
    m = re.match(r"(\d{2})-(\d{2})-(\d{4})$", date or "")
    return int(m.group(3) + m.group(2) + m.group(1)) if m else 0


def gold_case_date(title: str) -> str:
    m = re.search(r"\bon (\d{1,2} \w+,? \d{4})\s*$", title or "", flags=re.I)
    return norm_date(m.group(1)) if m else ""


def same_case(retrieved_name: str, retrieved_date: str, gold_title: str) -> bool:
    """Recall criterion: party-name overlap, relaxed when the dates agree."""
    g_left, g_right = split_parties(gold_title)
    r_left, r_right = split_parties(retrieved_name)
    overlap = jaccard(g_left | g_right, r_left | r_right)
    gd = gold_case_date(gold_title)
    same_date = bool(gd) and gd == norm_date(retrieved_date)
    return bool((same_date and overlap >= 0.25) or overlap >= 0.6)


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
def build_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for schema in (c.SC_SCHEMA, c.HC_SCHEMA):
        for doc in c.iter_feed_shards(schema):
            f = doc["fields"]
            key = (f.get("case_name", ""), f.get("judgment_date", ""))
            if schema == c.SC_SCHEMA and key in seen:
                continue  # one entry per SC judgment (first chunk)
            seen.add(key)
            entries.append(
                {
                    "id": doc["id"],
                    "schema": schema,
                    "case_name": f.get("case_name", ""),
                    "judgment_date": norm_date(f.get("judgment_date", "")),
                    "court": "Supreme Court of India" if schema == c.SC_SCHEMA else f.get("court_name", ""),
                    "citation": f.get("citations", ""),
                }
            )
        print(f"  {schema}: catalogue now {len(entries)} judgments", flush=True)
    with gzip.open(path, "wt", encoding="utf-8") as sink:
        json.dump(entries, sink, ensure_ascii=False)
    print(f"Case catalogue: {len(entries)} judgments -> {path}")
    return entries


@dataclass
class CaseCatalog:
    entries: list[dict[str, Any]]
    by_date: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    postings: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "CaseCatalog":
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                entries = json.load(handle)
        else:
            entries = build_catalog(path)
        cat = cls(entries)
        for i, e in enumerate(entries):
            left, right = split_parties(e["case_name"])
            e["_left"], e["_right"] = left, right
            e["_year"] = e["judgment_date"][-4:]
            cat.by_date[e["judgment_date"]].append(i)
            for token in (left | right) - _GENERIC:
                cat.postings[token].append(i)
        return cat

    def find_gold(self, gold_title: str) -> list[int]:
        """Corpus judgments matching a gold citation (for ceilings)."""
        out = []
        for i in self.by_date.get(gold_case_date(gold_title), []):
            e = self.entries[i]
            if same_case(e["case_name"], e["judgment_date"], gold_title):
                out.append(i)
        return out

    def resolve(self, left: set[str], right: set[str], year: str = "", before: int = 0) -> int | None:
        """Best corpus judgment for a cited 'left v. right' (optionally a year)."""
        distinctive = sorted((left | right) - _GENERIC, key=lambda t: len(self.postings.get(t, ())))
        distinctive = [t for t in distinctive if t in self.postings]
        if not distinctive or len(self.postings[distinctive[0]]) > 20000:
            return None
        best, best_score = None, 0.0
        for i in self.postings[distinctive[0]]:
            e = self.entries[i]
            if before and date_ord(e["judgment_date"]) > before:
                continue
            jl, jr = jaccard(left, e["_left"]), jaccard(right, e["_right"])
            if right and e["_right"]:
                if min(jl, jr) < 0.34 or max(jl, jr) < 0.5:
                    continue
                score = jl + jr
            else:
                score = 2 * jaccard(left | right, e["_left"] | e["_right"])
                if score < 1.2:
                    continue
            if year and e["_year"]:
                gap = int(year) - int(e["_year"])
                if gap < 0 or gap > 1:  # report year is the decision year or the next
                    continue
                score += 0.5
            score += 0.2 if e["schema"] == c.SC_SCHEMA else 0.0
            if score > best_score:
                best, best_score = i, score
        return best


# --------------------------------------------------------------------------
# "X v. Y" mentions in judgment text
# --------------------------------------------------------------------------
_PARTY = r"[A-Z][\w.&'/\-]*(?:,?\s+(?:[A-Z(][\w.&'/()\-]*|of|and|the|for|&|de|da)){0,9}"
_CITE_RE = re.compile(
    rf"(?P<left>{_PARTY})\s+(?:v\.|vs\.?|versus|v/s\.?|V\.|Vs\.?|VS\.?)\s+(?P<right>{_PARTY})"
    r"(?P<tail>[^\n]{0,60})"
)
_LEAD = re.compile(
    r"^(?:(?:In|See|Also|And|But|The|This|That|Further|Relying|Reliance|Placed|Upon|On|Of|Case|Cases|"
    r"Decision|Judgment|Judgement|Court|Held|Apex|Hon'ble|Honble|Supreme|Learned|Counsel|Where|"
    r"Similarly|Thus|Then|Accordingly|While|Referring|Followed|Following|Cited|Under|Re|As|"
    r"Per|Vide|Para|Paragraph|Reported|Recently|Earlier|Later|Again|Lordships)\b[\s,]*)+"
)


def extract_case_mentions(text: str, limit: int = 60) -> list[tuple[set[str], set[str], str]]:
    out, seen = [], set()
    for m in _CITE_RE.finditer(str(text or "")):
        left = _LEAD.sub("", m.group("left")).strip()
        right = re.split(r"\s+(?:\(supra\)|reported|wherein|where|which|in which|that|the court)\b",
                         m.group("right"), maxsplit=1)[0].strip(" ,.")
        lt, rt = party_tokens(left), party_tokens(right)
        if not lt or not rt:
            continue
        year = ""
        ym = re.search(r"[\[(]?\s*(1[89]\d\d|20[0-3]\d)\s*[\])]", m.group("tail")) or re.search(
            r"\b(1[89]\d\d|20[0-3]\d)\b", m.group("tail"))
        if ym:
            year = ym.group(1)
        key = (frozenset(lt), frozenset(rt))
        if key in seen:
            continue
        seen.add(key)
        out.append((lt, rt, year))
        if len(out) >= limit:
            break
    return out
