"""Shared runner for the RAG "full run" inference scripts (OpenRouter).

Differences from the original inference_*.py scripts:

* Context comes from the retrieved statutes/precedents in
  data/test_data/test_v2_rag2.json (cases 1-300), not the gold ones.
* No ground-truth fallbacks of any kind. If the model does not produce a
  section or a parseable prediction, the request is retried (with a larger
  token budget); if it still fails the field stays blank / None and the record
  is flagged in ``missing_fields``.
* The prompt is fitted to the input budget by clipping the retrieved context
  (statutes first to a fixed size, precedents share the remainder), never by
  cutting the tail of the prompt, so the output-format instructions survive.
* Cases are processed in dataset order (no random sampling), concurrently,
  with a resumable JSONL cache next to the output file.

The API key is read from OPENROUTER_API_KEY in .env or the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECTION_NAMES = (
    "Legal Issue Analysis",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Deliberation",
)
REQUIRED_FIELDS = (
    "Legal Issue Analysis",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Deliberation",
    "Explanation",
)
JSON_KEYS = {
    "legal_issue": "Legal Issue Analysis",
    "petitioner_arguments": "Arguments of Petitioner",
    "respondent_arguments": "Arguments of Respondent",
    "deliberation": "Deliberation",
    "explanation": "Explanation",
}
# Fields carried over from the dataset row into each output record. The gold
# fields are kept only so evaluators can match records; they are never read
# while building a prediction.
DROP_ROW_FIELDS = ("prompt",)


@dataclass
class ModelSpec:
    model: str                 # OpenRouter model id
    name: str                  # run name, e.g. "gpt_6_1_sol"
    title: str                 # human-readable name for X-Title
    build_prompt: Callable[[str, str, str], str]  # (facts, statutes, precedents)
    output_style: str          # "think" or "json" (informational)


# --------------------------------------------------------------------------- io

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def clip(text: Any, limit: int) -> str:
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


# ---------------------------------------------------------------------- context

def format_context(
    row: dict[str, Any], spec: ModelSpec, max_input_chars: int, statute_chars: int,
    min_precedent_chars: int,
) -> tuple[str, dict[str, Any]]:
    """Build the prompt, clipping retrieved context so it fits the budget."""
    facts = str(row.get("Simplified_Facts") or row.get("Facts") or "").strip()
    titles = list(row.get("section_titles") or [])
    texts = list(row.get("section_texts") or [])
    cases = list(row.get("cited_cases") or [])
    judgments = list(row.get("cited_case_judgments") or [])

    overhead = len(spec.build_prompt("", "", ""))
    available = max_input_chars - overhead
    if len(facts) > available * 0.6:          # pathological case: very long facts
        facts = clip(facts, int(available * 0.6))
    available -= len(facts)

    def statutes_block(limit: int) -> str:
        if not titles:
            return "No specific statutes provided."
        return "\n\n".join(
            f"Statute {i}. {t}:\n   {clip(texts[i - 1] if i - 1 < len(texts) else '', limit)}"
            for i, t in enumerate(titles, 1)
        )

    def precedents_block(limit: int) -> str:
        if not cases:
            return "No citations provided."
        return "\n\n".join(
            f"Precedent {i}. {c}:\n   "
            f"{clip(judgments[i - 1], limit) if i - 1 < len(judgments) and judgments[i - 1] else 'Details not available.'}"
            for i, c in enumerate(cases, 1)
        )

    s_limit = statute_chars
    statutes = statutes_block(s_limit)
    # leave room for precedents: statutes take at most 40% of what is left
    while len(statutes) > available * 0.4 and s_limit > 200:
        s_limit = int(s_limit * 0.8)
        statutes = statutes_block(s_limit)
    remaining = available - len(statutes)
    n_cases = max(1, len(cases))
    per_case = max(min_precedent_chars, remaining // n_cases - len("Precedent 10. ") - 150)
    precedents = precedents_block(per_case)
    while len(precedents) > remaining and per_case > 200:
        per_case = int(per_case * 0.9)
        precedents = precedents_block(per_case)

    prompt = spec.build_prompt(facts, statutes, precedents)
    info = {
        "prompt_chars": len(prompt),
        "statute_chars_each": s_limit,
        "precedent_chars_each": per_case,
        "n_statutes": len(titles),
        "n_precedents": len(cases),
    }
    return prompt, info


# ---------------------------------------------------------------------- parsing

_HEADING = r"(?:^|\n)[ \t>#*_\-]*(?:\d+[.)]\s*)?[*_]*\s*{name}\s*[*_]*\s*:?\s*[*_]*[ \t]*"
_STOP = (
    r"(?:\n[ \t>#*_\-]*(?:\d+[.)]\s*)?[*_]*\s*(?:{names})\s*[*_]*\s*(?::|\n)"
    r"|</think>|\n[ \t#*_]*PREDICTION|\n[ \t#*_]*EXPLANATION|$)"
)


def parse_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    names_alt = "|".join(re.escape(n) for n in SECTION_NAMES)
    for name in SECTION_NAMES:
        pattern = _HEADING.format(name=re.escape(name)) + r"(.*?)" + _STOP.format(names=names_alt)
        m = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL)
        value = m.group(1).strip() if m else ""
        value = re.sub(r"^[*_\s]+|[*_\s]+$", "", value)
        if re.fullmatch(r"\[.*\]", value or ""):   # unfilled template placeholder
            value = ""
        out[name] = value
    return out


def parse_json_payload(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


_PRED_PATTERNS = (
    r"#{1,3}\s*PREDICTION\s*:?\s*[*_`\[]*\s*([01])\b",
    r"[*_]*\s*PREDICTION\s*[*_]*\s*:\s*[*_`\[]*\s*([01])\b",
    r"[\"'](?:prediction|predicted_decision)[\"']\s*:\s*[\"']?([01])\b",
)


def parse_prediction(text: str, payload: dict[str, Any]) -> int | None:
    value = payload.get("prediction")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    for pattern in _PRED_PATTERNS:
        matches = list(re.finditer(pattern, text or "", re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
    return None


def parse_explanation(text: str) -> str:
    m = re.search(
        r"#{1,3}\s*EXPLANATION\s*:?\s*(.*?)(?=\n\s*#{1,3}\s*[A-Z]+\s*:|$)",
        text or "", re.IGNORECASE | re.DOTALL,
    )
    value = m.group(1).strip() if m else ""
    return "" if re.fullmatch(r"\[.*\]", value) else value


def parse_output(content: str, reasoning: str) -> dict[str, Any]:
    """Parse model output. Visible content first; provider reasoning only for
    sections that the content lacks (some models put the <think> block there)."""
    fields: dict[str, Any] = {name: "" for name in REQUIRED_FIELDS}
    payload = parse_json_payload(content)
    for key, name in JSON_KEYS.items():
        if isinstance(payload.get(key), str):
            fields[name] = payload[key].strip()
    sec_content = parse_sections(content)
    sec_reason = parse_sections(reasoning)
    for name in SECTION_NAMES:
        fields[name] = fields[name] or sec_content[name] or sec_reason[name]
    fields["Explanation"] = fields["Explanation"] or parse_explanation(content)
    prediction = parse_prediction(content, payload)
    source = "content"
    if prediction is None:
        prediction = parse_prediction(reasoning, {})
        source = "reasoning" if prediction is not None else None
    fields["Prediction"] = prediction
    fields["prediction_source"] = source
    fields["missing_fields"] = [n for n in REQUIRED_FIELDS if not fields[n]] + (
        ["Prediction"] if prediction is None else []
    )
    return fields


# ------------------------------------------------------------------------- http

def extract_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else str(b.get("text", ""))
            for b in content if isinstance(b, (str, dict))
        ).strip()
    return ""


def extract_reasoning(choice: dict[str, Any]) -> str:
    message = choice.get("message", {})
    value = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(value, str):
        return value.strip()
    details = message.get("reasoning_details")
    if isinstance(details, list):
        return "\n\n".join(
            str(d.get("summary") or d.get("text") or "").strip()
            for d in details if isinstance(d, dict)
        ).strip()
    return ""


class FatalAPIError(RuntimeError):
    """Account-level failure (bad key, no credit, key spend limit)."""


def check_key(api_key: str) -> None:
    """Fail before starting if the OpenRouter key has no credit left."""
    try:
        req = Request("https://openrouter.ai/api/v1/key",
                      headers={"Authorization": "Bearer " + api_key})
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8")).get("data") or {}
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise FatalAPIError(f"OpenRouter rejected the API key (HTTP {exc.code}).") from exc
        return
    except (URLError, TimeoutError, ValueError, OSError):
        return
    remaining = data.get("limit_remaining")
    if remaining is not None and remaining <= 0:
        raise FatalAPIError(
            f"OpenRouter key spend limit reached (used ${data.get('usage', 0):.2f} of "
            f"${data.get('limit')}). Raise it at https://openrouter.ai/settings/keys or use another key.")
    print(f"OpenRouter key: limit={data.get('limit')} remaining={remaining}", flush=True)


def request_completion(api_key: str, spec: ModelSpec, prompt: str, args: argparse.Namespace,
                       max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "usage": {"include": True},
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "X-Title": f"NyayaMind {spec.title} RAG full run",
        },
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            with urlopen(request, timeout=args.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            if body.get("choices"):
                return body
            last = RuntimeError(f"no choices: {str(body)[:500]}")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code in (401, 402, 403):
                raise FatalAPIError(f"OpenRouter HTTP {exc.code} (key/credit problem): {detail}") from exc
            if not (exc.code == 429 or exc.code >= 500):
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
            last = RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}")
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as exc:
            last = exc
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"OpenRouter request failed after retries: {last}")


# ------------------------------------------------------------------------- run

def run_case(api_key: str, spec: ModelSpec, row: dict[str, Any], prompt: str,
             args: argparse.Namespace) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    for attempt in range(1, args.max_attempts + 1):
        max_tokens = min(args.max_new_tokens * attempt, args.max_new_tokens_cap)
        try:
            body = request_completion(api_key, spec, prompt, args, max_tokens)
        except FatalAPIError:
            raise
        except RuntimeError as exc:
            attempts.append({"attempt": attempt, "error": str(exc)})
            continue
        choice = body["choices"][0]
        content = extract_content(choice.get("message", {}))
        reasoning = extract_reasoning(choice)
        usage = body.get("usage") or {}
        for k in usage_total:
            usage_total[k] += usage.get(k) or 0
        parsed = parse_output(content, reasoning)
        candidate = {
            **parsed,
            "raw_output": content,
            "raw_reasoning": reasoning,
            "finish_reason": choice.get("finish_reason"),
        }
        attempts.append({
            "attempt": attempt, "max_tokens": max_tokens,
            "finish_reason": choice.get("finish_reason"),
            "missing_fields": parsed["missing_fields"],
            "completion_tokens": usage.get("completion_tokens"),
        })
        if best is None or len(candidate["missing_fields"]) < len(best["missing_fields"]):
            best = candidate
        if not best["missing_fields"] or choice.get("finish_reason") == "content_filter":
            break
    if best is None:
        best = {name: "" for name in REQUIRED_FIELDS}
        best.update({"Prediction": None, "prediction_source": None,
                     "missing_fields": list(REQUIRED_FIELDS) + ["Prediction"],
                     "raw_output": "", "raw_reasoning": "", "finish_reason": None,
                     "error": attempts[-1].get("error") if attempts else "no attempts"})
    record = {k: v for k, v in row.items() if k not in DROP_ROW_FIELDS}
    record.update({
        "Legal Issue Analysis": best["Legal Issue Analysis"],
        "Arguments of Petitioner": best["Arguments of Petitioner"],
        "Arguments of Respondent": best["Arguments of Respondent"],
        "Deliberation": best["Deliberation"],
        "Prediction": best["Prediction"],
        "Explanation": best["Explanation"],
        "predicted_decision": best["Prediction"],
        "prediction_source": best["prediction_source"],
        "missing_fields": best["missing_fields"],
        "complete": not best["missing_fields"],
        "raw_output": best["raw_output"],
        "raw_reasoning": best["raw_reasoning"][: args.keep_reasoning_chars],
        "finish_reason": best["finish_reason"],
        "attempts": attempts,
        "usage": usage_total,
        "error": best.get("error", ""),
    })
    return record


def parse_args(spec: ModelSpec) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"RAG full run with {spec.title} via OpenRouter "
                                            "(retrieved context, no gold fallbacks).")
    p.add_argument("--dataset", type=Path, default=Path("data/test_data/test_v2_rag2.json"))
    p.add_argument("--output", type=Path, default=Path(f"logs/{spec.name}_full_run.json"))
    p.add_argument("--start", type=int, default=0, help="0-based first case (default 0 = case 1)")
    p.add_argument("--limit", type=int, default=300, help="number of cases (default 300)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-seq-length", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens-cap", type=int, default=16384,
                   help="token budget ceiling for retries (attempt k uses k * max-new-tokens)")
    p.add_argument("--statute-chars", type=int, default=1500)
    p.add_argument("--min-precedent-chars", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--reasoning-effort", default=None,
                   help="optional OpenRouter reasoning effort (low/medium/high)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--max-retries", type=int, default=5, help="HTTP retries per request")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="generation attempts per case when output is incomplete")
    p.add_argument("--keep-reasoning-chars", type=int, default=20000)
    p.add_argument("--rerun-incomplete", action="store_true",
                   help="re-request cached cases that are incomplete or errored")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main(spec: ModelSpec) -> int:
    args = parse_args(spec)
    load_dotenv(PROJECT_ROOT / ".env")
    for attr in ("dataset", "output"):
        path = getattr(args, attr)
        if not path.is_absolute():
            setattr(args, attr, PROJECT_ROOT / path)
    if args.max_seq_length <= args.max_new_tokens:
        raise ValueError("--max-seq-length must exceed --max-new-tokens")

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = [dict(row, case_index=args.start + i)
            for i, row in enumerate(rows[args.start: args.start + args.limit])]
    max_input_chars = (args.max_seq_length - args.max_new_tokens) * 4
    prompts: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for row in rows:
        prompt, info = format_context(row, spec, max_input_chars, args.statute_chars,
                                      args.min_precedent_chars)
        prompts.append((row, prompt, info))
    sizes = [info["prompt_chars"] for _, _, info in prompts]
    print(f"{spec.title}: {len(rows)} cases (index {args.start}-{args.start + len(rows) - 1}) "
          f"from {args.dataset.name}; prompt chars mean={sum(sizes) // max(1, len(sizes))} "
          f"max={max(sizes, default=0)} budget={max_input_chars}", flush=True)
    if args.dry_run:
        print(prompts[0][1][:3000] if prompts else "")
        print("...\n" + (prompts[0][1][-1500:] if prompts else ""))
        return 0

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing (.env or environment)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output.with_suffix(".jsonl")
    if args.no_resume and cache_path.exists():
        cache_path.unlink()
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[str(rec.get("case_index"))] = rec
    todo = []
    for row, prompt, info in prompts:
        rec = cache.get(str(row["case_index"]))
        # API errors are never model output, so errored cases are always re-requested.
        if rec is None or rec.get("error") or (args.rerun_incomplete and not rec.get("complete")):
            todo.append((row, prompt, info))
    print(f"cached={len(prompts) - len(todo)} to_run={len(todo)} workers={args.workers}", flush=True)

    if todo:
        check_key(api_key)
    lock = threading.Lock()
    fatal: FatalAPIError | None = None
    done = 0

    def work(item):
        row, prompt, info = item
        rec = run_case(api_key, spec, row, prompt, args)
        rec["prompt_info"] = info
        return rec

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, \
            cache_path.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(work, item) for item in todo]
        for future in as_completed(futures):
            try:
                rec = future.result()
            except FatalAPIError as exc:
                fatal = exc
                for pending in futures:
                    pending.cancel()
                continue
            with lock:
                cache[str(rec.get("case_index"))] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                print(f"[{done}/{len(todo)}] case {rec['case_index'] + 1} {rec.get('document_id')} pred={rec['Prediction']} "
                      f"missing={rec['missing_fields']} attempts={len(rec['attempts'])} "
                      f"cost=${rec['usage']['cost']:.4f}", flush=True)

    results = [cache[str(row["case_index"])] for row, _, _ in prompts
               if str(row["case_index"]) in cache]
    cost = sum((r.get("usage") or {}).get("cost") or 0 for r in results)
    summary = {
        "n_cases": len(results),
        "complete": sum(bool(r.get("complete")) for r in results),
        "prediction_missing": sum(r.get("Prediction") is None for r in results),
        "errors": sum(bool(r.get("error")) for r in results),
        "missing_by_field": {
            name: sum(name in (r.get("missing_fields") or []) for r in results)
            for name in list(REQUIRED_FIELDS) + ["Prediction"]
        },
        "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in results),
        "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in results),
        "cost_usd": round(cost, 4),
    }
    envelope = {
        "run_name": f"{spec.name}_full_run",
        "model_path": spec.model,
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT)) if args.dataset.is_relative_to(PROJECT_ROOT) else str(args.dataset),
        "case_range": [args.start + 1, args.start + len(rows)],
        "context": "retrieved statutes and precedents (Vespa RAG v2), not gold",
        "gold_fallback_disabled": True,
        "output_style": spec.output_style,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary,
        "output_format": (
            "Each result holds the dataset row plus model-generated Legal Issue Analysis, "
            "Arguments of Petitioner/Respondent, Deliberation, Prediction, Explanation. "
            "Missing outputs stay blank/None and are listed in missing_fields; ground-truth "
            "fields are never used as fallbacks."
        ),
        "results": results,
    }
    args.output.write_text(json.dumps(envelope, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if fatal is not None:
        print(f"ABORTED: {fatal}\nRe-run the same command after fixing the key; "
              "finished cases are cached and errored ones are retried.", file=sys.stderr)
        return 3
    return 0
