"""Explicit statute references in case facts, resolved against the corpus.

Case facts often name the provisions in dispute outright ("benefits under
Sections 80HH and 80IA of the Income Tax Act, 1961", "Section 3 of the 1971
Act").  Dense/BM25 search over a long facts query routinely misses these, so
this module

* builds (once, cached) a catalogue of every statute section in the Central
  and State Act collections from the offline feed shards
  (``data/vespa_feed/_statute_catalog.json.gz``);
* extracts ``Section/Article/Rule <n> of <Act>`` references from free text,
  resolving short forms ("the 1971 Act", "the said Act", "IPC", "CrPC");
* maps each reference to the matching corpus sections.

Only the query text (the case facts) is ever parsed - never gold citations.
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

CATALOG_PATH = c.FEED_DIR / "_statute_catalog.json.gz"
TEXT_CAP = 8000


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
_ACT_NOISE = frozenset("the of and a an in for to".split())
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}


def act_words(name: str) -> tuple[str, ...]:
    """'The Income-tax Act, 1961' -> ('income', 'tax', 'act', '1961')."""
    text = c.preprocess_text(str(name or "").replace("_", " "))
    text = re.sub(r"\(.*?\)", " ", text) if len(text) > 60 else text  # drop long parentheticals
    words = re.findall(r"[a-z]+|\d{4}", text)
    # light plural folding: "Industrial Dispute Act" == "Industrial Disputes Act"
    return tuple(
        (w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
        for w in words
        if w not in _ACT_NOISE
    )


def act_year(words: tuple[str, ...]) -> str:
    years = [w for w in words if w.isdigit() and len(w) == 4]
    return years[-1] if years else ""


def norm_number(number: str) -> str:
    """'80-IA.' -> '80ia'; '2.0' -> '2'; '13(1)(d)' -> '13'."""
    text = re.sub(r"^(\d+)\.0$", r"\1", str(number or "").strip())
    text = re.split(r"[(\[]", text, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", text.lower())


def norm_title_key(title: str) -> str:
    """Key for 'Section 80IA in The Income Tax Act, 1961' style titles."""
    match = re.match(r"\s*(?:section|article|rule)\s+(.*?)\s+in\s+(.*)$", title or "", flags=re.I)
    if not match:
        return "|" + " ".join(act_words(title))
    return f"{norm_number(match.group(1))}|{' '.join(act_words(match.group(2)))}"


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
def build_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for schema in [c.CA_SCHEMA, *c.sa_schemas()]:
        for doc in c.iter_feed_shards(schema):
            f = doc["fields"]
            entries.append(
                {
                    "id": doc["id"],
                    "schema": schema,
                    "name_of_statute": f.get("name_of_statute", ""),
                    "section_number": f.get("section_number", ""),
                    "section_title": f.get("section_title", ""),
                    "state_name": f.get("state_name", ""),
                    "section_text": (f.get("section_text") or "")[:TEXT_CAP],
                }
            )
    with gzip.open(path, "wt", encoding="utf-8") as sink:
        json.dump(entries, sink, ensure_ascii=False)
    print(f"Statute catalogue: {len(entries)} sections -> {path}")
    return entries


@dataclass
class Catalog:
    entries: list[dict[str, Any]]
    acts: dict[tuple[str, ...], list[int]] = field(default_factory=lambda: defaultdict(list))
    by_year: dict[str, set[tuple[str, ...]]] = field(default_factory=lambda: defaultdict(set))
    sections: dict[tuple[tuple[str, ...], str], list[int]] = field(default_factory=lambda: defaultdict(list))
    keys: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "Catalog":
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                entries = json.load(handle)
        else:
            entries = build_catalog(path)
        cat = cls(entries)
        for i, e in enumerate(entries):
            words = act_words(e["name_of_statute"])
            e["_act"] = words
            e["_num"] = norm_number(e["section_number"])
            cat.acts[words].append(i)
            cat.by_year[act_year(words)].add(words)
            cat.sections[(words, e["_num"])].append(i)
            cat.keys.add(f"{e['_num']}|{' '.join(words)}")
        return cat

    # -- act matching ------------------------------------------------------
    def match_act(self, mention: tuple[str, ...]) -> list[tuple[str, ...]]:
        """Corpus Acts that a mentioned Act name refers to (best first)."""
        if mention in self.acts:
            return [mention]
        year = act_year(mention)
        name = set(w for w in mention if not w.isdigit())
        if not name:
            return []
        pool = self.by_year.get(year, set()) if year else self.acts.keys()
        scored = []
        for words in pool:
            other = set(w for w in words if not w.isdigit())
            if not other:
                continue
            inter = len(name & other)
            jacc = inter / len(name | other)
            if year:
                score = max(jacc, 0.9 * inter / len(name))
                ok = score >= 0.6
            else:  # every mentioned word must appear; prefer the closest-length name
                score = jacc if inter == len(name) else 0.0
                ok = score >= 0.4
            if ok:
                scored.append((score, -len(other), words))
        scored.sort(reverse=True)
        if not scored:
            return []
        best = scored[0][0]
        return [w for s, _, w in scored if s >= best - 1e-9][:4]

    def lookup(self, act: tuple[str, ...], number: str) -> list[int]:
        return self.sections.get((act, number), [])

    def has_title(self, title: str) -> bool:
        return norm_title_key(title) in self.keys


# --------------------------------------------------------------------------
# Reference extraction
# --------------------------------------------------------------------------
ABBREVIATIONS = {
    r"i\.?\s?p\.?\s?c\.?": "Indian Penal Code, 1860",
    r"cr\.?\s?p\.?\s?c\.?": "Code of Criminal Procedure, 1973",
    r"c\.?\s?p\.?\s?c\.?": "Code of Civil Procedure, 1908",
    r"n\.?\s?d\.?\s?p\.?\s?s\.?(?:\s?act)?": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    r"n\.?\s?i\.?\s?act": "Negotiable Instruments Act, 1881",
    r"m\.?\s?v\.?\s?act": "Motor Vehicles Act, 1988",
    r"i\.?\s?t\.?\s?act": "Income Tax Act, 1961",
    r"p\.?\s?c\.?\s?act": "Prevention of Corruption Act, 1988",
    r"i\.?\s?d\.?\s?act": "Industrial Disputes Act, 1947",
    r"bns": "Bharatiya Nyaya Sanhita, 2023",
    r"bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    r"evidence act": "Indian Evidence Act, 1872",
    r"constitution(?: of india)?": "Constitution of India",
    r"cst act": "Central Sales Tax Act, 1956",
}
# Abbreviations that are ordinary words too: only honoured right after "Section/Article .. of".
_TAIL_ONLY = {r"constitution(?: of india)?", r"evidence act"}
_ABBR_RE = re.compile(r"^(?:" + "|".join(f"(?P<a{i}>{p})" for i, p in enumerate(ABBREVIATIONS)) + r")$", re.I)
_ABBR_NAMES = list(ABBREVIATIONS.values())

# "Section 3", "Sections 80HH and 80IA", "Sec. 302/34", "S. 13(1)(d)", "Article 226"
_NUM = r"\d+[A-Z]{0,3}(?:-[A-Z]{1,3})?(?:\s?\([0-9a-zA-Z]{1,4}\))*"
_NUM_LIST = rf"{_NUM}(?:\s*(?:,\s*and|,\s*or|,|/|and|&|or|to|read with|r/w)\s*(?:sections?|secs?\.?|s\.)?\s*{_NUM})*"
_REF_RE = re.compile(
    rf"\b(?P<kind>sections?|secs?\.|sec|ss\.|s\.|u/s\.?|articles?|arts?\.|rules?)\s*(?P<nums>{_NUM_LIST})",
    re.I,
)
# What follows the numbers: "of the Income Tax Act, 1961" / "of IPC" / "of the 1971 Act"
_ACT_TAIL = re.compile(
    r"^\s*(?:,\s*)?(?:of|under|in)\s+(?:the\s+)?(?:(?:adopted|amended|present|repealed|erstwhile|unamended)\s+)?(?P<act>"
    r"(?:said|same|aforesaid|principal|amended|new|old)\s+(?:act|code|rules)"
    r"|\d{4}\s+(?:act|code|rules)"
    r"|[A-Z][\w().,'&\- ]{2,160}?(?:Act|Code|Sanhita|Adhiniyam|Rules|Order|Constitution of India)(?:,?\s*\d{4})?"
    r"|[A-Za-z.\s]{2,12}?(?=[\s,.;)]|$)"
    r")",
)
# Full Act names anywhere in the text: "The Kerala Private Forests (Vesting and Assignment) Act, 1971"
_ACT_NAME_RE = re.compile(
    r"(?:[Tt]he\s+)?(?P<name>[A-Z][A-Za-z().,'&\- ]{3,160}?\b(?:Act|Code|Sanhita|Adhiniyam)),?\s*(?P<year>\d{4})"
)


@dataclass
class Reference:
    kind: str  # section | article | rule
    number: str
    act_mention: str
    acts: list[tuple[str, ...]]


def _numbers(nums: str) -> list[str]:
    out = []
    for token in re.findall(_NUM, nums):
        number = norm_number(token)
        if number and number not in out:
            out.append(number)
    # "Sections 3 to 5" -> 3, 4, 5
    span = re.search(r"(\d+)\s*to\s*(\d+)", nums)
    if span and int(span.group(2)) - int(span.group(1)) <= 10:
        out += [str(n) for n in range(int(span.group(1)), int(span.group(2)) + 1) if str(n) not in out]
    return out


def _abbr(text: str) -> str:
    match = _ABBR_RE.match(text.strip().rstrip(".,"))
    if match:
        return _ABBR_NAMES[int(match.lastgroup[1:])]
    return ""


def extract_references(text: str, catalog: Catalog) -> tuple[list[Reference], list[tuple[str, ...]]]:
    """Return (section references, Acts named in the text) resolved against the corpus."""
    text = str(text or "")
    # Acts named in full, in order of appearance (for "the 1971 Act", "the said Act").
    named: list[tuple[int, tuple[str, ...], str]] = []
    for m in _ACT_NAME_RE.finditer(text):
        name = re.sub(r"^(?:under|of|in|by|and|the|that|as|per)\s+", "", m.group("name").strip(), flags=re.I)
        words = act_words(f"{name} {m.group('year')}")
        if len(words) >= 3:
            named.append((m.start(), words, m.group("year")))
    for abbr, full in ABBREVIATIONS.items():
        if abbr in _TAIL_ONLY:
            continue
        for m in re.finditer(rf"(?<![A-Za-z]){abbr}(?![A-Za-z])", text, flags=re.I):
            if abbr in ("bns", "bnss") and not m.group(0).isupper():
                continue
            named.append((m.start(), act_words(full), act_year(act_words(full))))
    named.sort()

    def resolve(mention: str, position: int) -> list[tuple[str, ...]]:
        mention = mention.strip().rstrip(".,;")
        low = mention.lower()
        if not mention:
            return []
        full = _abbr(mention)
        if full:
            return catalog.match_act(act_words(full))
        year = re.fullmatch(r"(\d{4})\s+(?:act|code|rules)", low)
        if year:  # "the 1971 Act"
            for _, words, y in reversed(named):
                if y == year.group(1):
                    return catalog.match_act(words)
            return []
        if re.fullmatch(r"(?:said|same|aforesaid|principal|amended|new|old)\s+(?:act|code|rules)", low):
            before = [w for p, w, _ in named if p < position]
            return catalog.match_act(before[-1]) if before else []
        words = act_words(mention)
        if len(words) < 2:
            return []
        if not act_year(words):  # "the Income Tax Act" -> use the year of a full mention if any
            for _, w, _ in named:
                if set(words) - {"act"} <= set(w):
                    return catalog.match_act(w)
        return catalog.match_act(words)

    refs: list[Reference] = []
    seen: set[tuple[str, str]] = set()
    for m in _REF_RE.finditer(text):
        kind_raw = m.group("kind").lower()
        kind = "article" if kind_raw.startswith("art") else "rule" if kind_raw.startswith("rule") else "section"
        tail = _ACT_TAIL.match(text[m.end(): m.end() + 200])
        mention = tail.group("act") if tail else ""
        acts = resolve(mention, m.start()) if mention else []
        if not acts and not mention:
            # "Section 4 ... " with no Act: assume the closest Act named before it in the same paragraph.
            before = [(p, w) for p, w, _ in named if 0 <= m.start() - p <= 600]
            acts = catalog.match_act(before[-1][1]) if before else []
        if kind == "article" and not acts:
            acts = catalog.match_act(act_words("Constitution of India"))
        for number in _numbers(m.group("nums")):
            key = (number, "|".join(" ".join(a) for a in acts) or mention.lower())
            if key in seen:
                continue
            seen.add(key)
            refs.append(Reference(kind, number, mention, acts))

    acts_named: list[tuple[str, ...]] = []
    for _, words, _ in named:
        for act in catalog.match_act(words):
            if act not in acts_named:
                acts_named.append(act)
    return refs, acts_named


def resolve_sections(refs: list[Reference], catalog: Catalog) -> list[int]:
    """Catalogue indices of the sections the references point to (in order, deduped)."""
    out: list[int] = []
    for ref in refs:
        for act in ref.acts:
            hits = catalog.lookup(act, ref.number)
            if hits:
                # prefer the Central Act copy, then the first chunk
                hits = sorted(hits, key=lambda i: (catalog.entries[i]["schema"] != c.CA_SCHEMA, i))
                if hits[0] not in out:
                    out.append(hits[0])
                break
    return out
