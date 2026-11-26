"""LLM-as-a-judge evaluation of generated legal analyses.

Judge backends: the Anthropic API (default, ``claude-haiku-5-5``; key from
ANTHROPIC_API_KEY) or OpenRouter (any other ``--model`` slug; key from
OPENROUTER_API_KEY).  The judge is blind: it never sees which system wrote the
text, and model / vendor names inside generated text are masked.

Three judge prompts are sent per case.  They follow the
Content-Coverage / Faithfulness / Narrative-Quality protocol, adapted from
"introduction vs. paper metadata" to "legal analysis vs. case record":

1. Content Coverage & Consistency  - generated analysis vs. the actual
   reasoning of the court (``Simplified_Reasoning`` by default).
2. Faithfulness & Citation Context Quality - generated analysis vs. the
   material the model was given: the case facts, the statutes and the cited
   precedents (taken from the generated record, i.e. exactly what the
   inference prompt contained, or from ``--context``).
3. Narrative Quality - the generated analysis alone.

Scores are floats in [0, 1].  Citation Context Quality is null when the
analysis cites no statute or precedent; such cases are excluded from its mean
and counted in summary.json.

Example:

    python code/evaluation/evaluate_llm_judge.py \
        --generated logs/claude_opus_5_5_350.json \
        --ground-truth data/test_data/test_v2.json \
        --output-dir evaluation_results/claude_opus_5_5_judge

Outputs (same layout as evaluate_text_metrics.py, plus raw judgments):

* ``judgments.jsonl``: raw judge replies with rationales (resumable cache).
* ``per_example.csv``: the five scores per case.
* ``cumulative.csv``: running means.
* ``summary.json``: means, counts, judge model, configuration.

Keys are read from .env or the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_text_metrics import iter_json_array, record_key, text_value  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "claude-haiku-5-5"
# Anthropic list prices, USD per 1M tokens (input, output); Haiku 5.5 prompts > 100K tokens cost 5x.
ANTHROPIC_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}

SYSTEM_PROMPT = ("You are an impartial expert evaluator of legal analyses of Indian court appeals. "
                 "The text you assess was written by an anonymous system; judge only the text against the "
                 "material given, using the rubric. Reply with the requested JSON only.")

# Blinding: mask any model / vendor self-reference in generated text before the judge sees it.
BLIND_PATTERN = re.compile(
    r"\b(?:claude(?:[\s-]*(?:opus|sonnet|haiku|fable))?(?:[\s-]*\d+(?:\.\d+)?)?|anthropic|"
    r"chat\s*gpt|gpt[\s-]*\d+(?:\.\d+)?(?:[\s-]*sol)?|openai|nemotron(?:[\s-]*\d+)?(?:[\s-]*ultra)?|nvidia|"
    r"deepseek(?:[\s-]*r\d)?|qwen\d*|glm[\s-]*\d*(?:\.\d+)?|llama[\s-]*\d*|gemini|mistral)\b",
    re.IGNORECASE)


def blind(text: str) -> str:
    return BLIND_PATTERN.sub("[system]", text)


def _schema(fields: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


_S, _N = {"type": "string"}, {"type": "number"}
SCHEMAS = {  # rationale before score, so the judge justifies before it grades
    "coverage": _schema({"Content Coverage Rationale": _S, "Content Coverage": _N,
                         "Consistency Rationale": _S, "Consistency": _N}),
    "faithfulness": _schema({"Faithfulness Rationale": _S, "Faithfulness": _N,
                             "Citation Context Quality Rationale": _S,
                             "Citation Context Quality": {"anyOf": [_N, {"type": "null"}]}}),
    "narrative": _schema({"Narrative Quality Rationale": _S, "Narrative Quality": _N}),
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS = (
    "content_coverage",
    "consistency",
    "faithfulness",
    "citation_context_quality",
    "narrative_quality",
)
JUDGE_KEYS = {
    "content_coverage": "Content Coverage",
    "consistency": "Consistency",
    "faithfulness": "Faithfulness",
    "citation_context_quality": "Citation Context Quality",
    "narrative_quality": "Narrative Quality",
}

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
COVERAGE_PROMPT = '''Content Coverage & Consistency
{pl}:
""" {pred} """
{rl}:
""" {label} """
You will evaluate a {pl} of an Indian court appeal against the {rl}
and output ONLY this JSON:
{{ "Content Coverage": float (0.0-1.0),
"Content Coverage Rationale": "bullet-style justification",
"Consistency": float (0.0-1.0),
"Consistency Rationale": "bullet-style justification" }}
-
Scale: 0.0 = no alignment, 1.0 = perfect match
Content Coverage checks how many core legal issues, findings, applied provisions and supporting facts
from the {rl} appear in the {pl}.
Consistency checks for hallucinations or internal contradictions in the {pl},
including claims that contradict the {rl}. Partial scores are allowed.'''

FAITHFULNESS_PROMPT = '''Faithfulness & Citation Context Quality
{pl}:
""" {pred} """
Metadata:
Case Name: {case_name}
Case Facts:
""" {facts} """
Statutes provided:
""" {statutes} """
Cited precedents provided:
""" {precedents} """
You are evaluating a {pl} against its Metadata and output
ONLY this JSON:
{{ "Faithfulness": float (0.0-1.0),
"Faithfulness Rationale": "bullet-style justification",
"Citation Context Quality": float (0.0-1.0) or null,
"Citation Context Quality Rationale": "bullet-style justification" }}
Faithfulness checks if each claim is supported by the Case Facts, the Statutes provided, or the
Cited precedents provided (claims of law that are widely established may count as supported;
invented facts, parties, dates, amounts or holdings may not).
Citation Context Quality checks if sentences that cite or rely on a statute or precedent accurately
reflect the cited metadata (correct provision, correct holding, relevant to the point made).
Use null for Citation Context Quality only if the analysis cites no statute or precedent at all.
Scale: 0.0 = unsupported / inaccurate, 1.0 = fully supported / accurate. Partial scores are allowed.'''

NARRATIVE_PROMPT = '''Narrative Quality
{pl}:
""" {pred} """
Scale: 0.0 = disfluent, 1.0 = highly fluent and readable
Output ONLY JSON:
{{ "Narrative Quality": float (0.0-1.0),
"Narrative Quality Rationale": "bullet-style justification" }}
Evaluation includes grammar, logical flow, formal legal tone, and readability. Deduct points for
issues (e.g. rambling, repetition, meta-commentary about the task, truncation); provide
justification in bullet style.'''


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
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


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
    return [str(value)] if value else []


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def format_metadata(record: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    titles = as_list(record.get("section_titles"))
    texts = as_list(record.get("section_texts"))
    statutes = "\n".join(
        f"{i}. {title}: {clip(texts[i - 1] if i - 1 < len(texts) else '', args.max_item_chars)}"
        for i, title in enumerate(titles[: args.max_items], 1)
    ) or "None provided."
    cases = as_list(record.get("cited_cases"))
    judgments = as_list(record.get("cited_case_judgments"))
    precedents = "\n".join(
        f"{i}. {title}"
        + (f": {clip(judgments[i - 1], args.max_item_chars)}" if args.include_judgments and i - 1 < len(judgments) else "")
        for i, title in enumerate(cases[: args.max_items], 1)
    ) or "None provided."
    return {
        "case_name": str(record.get("Case Name", "")),
        "facts": clip(record.get("Simplified_Facts") or record.get("Facts") or "", args.max_facts_chars),
        "statutes": statutes,
        "precedents": precedents,
    }


def parse_judge_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text[:300]!r}")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # tolerate unquoted rationale text / trailing commas
        cleaned = re.sub(r",\s*}", "}", raw)
        return json.loads(cleaned)


def score_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None or (isinstance(value, str) and value.strip().lower() in {"null", "none", "n/a", ""}):
        return None
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if not m:
            return None
        value = m.group(0)
    number = float(value)
    if number > 1.0 and number <= 10.0:  # judge answered on a 0-10 scale
        number /= 10.0
    return max(0.0, min(1.0, number))


class FatalAPIError(RuntimeError):
    """Account-level failure (bad key, no credit, key spend limit): retrying is pointless."""


def check_key(api_key: str) -> None:
    """Fail before judging if the OpenRouter key has no credit left."""
    try:
        req = Request("https://openrouter.ai/api/v1/key",
                      headers={"Authorization": "Bearer " + api_key})
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8")).get("data") or {}
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise FatalAPIError(f"OpenRouter rejected the API key (HTTP {exc.code}).") from exc
        return
    except (URLError, TimeoutError, ValueError):
        return  # preflight is best-effort
    remaining = data.get("limit_remaining")
    if remaining is not None and remaining <= 0:
        raise FatalAPIError(
            f"OpenRouter key spend limit reached (used ${data.get('usage', 0):.2f} of "
            f"${data.get('limit')}). Raise the key limit at https://openrouter.ai/settings/keys "
            "or put a different key in OPENROUTER_API_KEY, then re-run (completed judgments are cached).")
    print(f"OpenRouter key: limit={data.get('limit')} remaining={remaining}", flush=True)


def is_anthropic(model: str) -> bool:
    return model.startswith("claude-")


def check_anthropic_key(client: Any, model: str) -> None:
    """Free preflight: the key is valid and can use the judge model."""
    import anthropic
    try:
        info = client.models.retrieve(model)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise FatalAPIError(f"Anthropic rejected the API key ({exc.status_code}).") from exc
    except anthropic.NotFoundError as exc:
        raise FatalAPIError(f"Judge model {model} is not available to this key.") from exc
    print(f"Anthropic key OK; judge model {info.id}", flush=True)


class Judge:
    def __init__(self, api_key: str, args: argparse.Namespace):
        self.api_key = api_key
        self.args = args
        self.client = None
        if is_anthropic(args.model):
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, max_retries=args.max_retries,
                                              timeout=float(args.timeout))

    def call(self, prompt: str, kind: str = "coverage") -> tuple[dict[str, Any], dict[str, Any]]:
        if self.client is not None:
            return self.call_anthropic(prompt, kind)
        return self.call_openrouter(prompt)

    def call_anthropic(self, prompt: str, kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
        import anthropic
        request: dict[str, Any] = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMAS[kind]}},
        }
        if self.args.reasoning_effort == "none":
            request["thinking"] = {"type": "disabled"}
        elif self.args.reasoning_effort != "default":
            request["output_config"]["effort"] = self.args.reasoning_effort
        last_error: Exception | None = None
        for _ in range(3):  # SDK already retries 429/5xx/connection errors; this covers bad replies
            try:
                response = self.client.messages.create(**request)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                raise FatalAPIError(f"Anthropic HTTP {exc.status_code}: {exc.message}") from exc
            except anthropic.BadRequestError as exc:
                if "credit" in str(exc.message).lower() or "billing" in str(exc.message).lower():
                    raise FatalAPIError(f"Anthropic credit/billing problem, aborting: {exc.message}") from exc
                raise RuntimeError(f"Anthropic HTTP 400: {exc.message}") from exc
            except anthropic.APIStatusError as exc:
                last_error = RuntimeError(f"Anthropic HTTP {exc.status_code}: {exc.message}")
                continue
            except anthropic.APIConnectionError as exc:
                last_error = exc
                continue
            usage = response.usage
            prompt_tokens, completion_tokens = usage.input_tokens or 0, usage.output_tokens or 0
            price_in, price_out = ANTHROPIC_PRICES.get(self.args.model, (0.0, 0.0))
            if self.args.model == "claude-haiku-5-5" and prompt_tokens > 100_000:
                price_in, price_out = price_in * 5, price_out * 5
            usage_row = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "cost": (prompt_tokens * price_in + completion_tokens * price_out) / 1e6}
            if response.stop_reason == "refusal":
                category = getattr(response.stop_details, "category", None) if response.stop_details else None
                raise RuntimeError(f"judge refused (category={category})")
            if response.stop_reason == "max_tokens":
                last_error = RuntimeError("judge reply truncated at max_tokens")
                continue
            text = next((b.text for b in response.content if b.type == "text"), "")
            try:
                return parse_judge_json(text), usage_row
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"judge call failed: {last_error}")

    def call_openrouter(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.args.seed is not None:
            payload["seed"] = self.args.seed
        if self.args.reasoning_effort != "default":
            payload["reasoning"] = (
                {"enabled": False} if self.args.reasoning_effort == "none"
                else {"effort": self.args.reasoning_effort}
            )
        request = Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "X-Title": "NyayaMind LLM-judge evaluation",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.args.max_retries + 1):
            try:
                with urlopen(request, timeout=self.args.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"].get("content") or ""
                return parse_judge_json(content), body.get("usage") or {}
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}")
                if exc.code in (401, 402, 403):
                    raise FatalAPIError(
                        f"OpenRouter HTTP {exc.code} (key/credit problem, aborting): {detail}") from exc
                if exc.code not in (408, 429) and exc.code < 500:
                    break
            except (URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"judge call failed: {last_error}")

    def evaluate(self, key: str, pred: str, label: str, record: dict[str, Any]) -> dict[str, Any]:
        meta = format_metadata(record, self.args)
        out: dict[str, Any] = {"key": key, "model": self.args.model, "usage": {}, "errors": {}}
        pred = blind(pred)
        labels = {"pl": self.args.pred_label, "rl": self.args.ref_label}
        jobs = {
            "coverage": COVERAGE_PROMPT.format(pred=pred, label=label, **labels) if label else None,
            "faithfulness": FAITHFULNESS_PROMPT.format(pred=pred, **meta, **labels),
            "narrative": NARRATIVE_PROMPT.format(pred=pred, **labels),
        }
        for name, prompt in jobs.items():
            if prompt is None:
                continue
            try:
                payload, usage = self.call(prompt, name)
                out[name] = payload
                out["usage"][name] = usage
            except FatalAPIError:
                raise
            except Exception as exc:  # noqa: BLE001 - record and continue
                out["errors"][name] = str(exc)
        scores = {}
        for metric, judge_key in JUDGE_KEYS.items():
            source = {"content_coverage": "coverage", "consistency": "coverage",
                      "faithfulness": "faithfulness", "citation_context_quality": "faithfulness",
                      "narrative_quality": "narrative"}[metric]
            try:
                scores[metric] = score_value(out.get(source) or {}, judge_key)
            except (TypeError, ValueError):
                scores[metric] = None
        out["scores"] = scores
        return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path, default=None,
                        help="Dataset holding the statutes/precedents given to the model "
                             "(default: the fields stored in the generated records).")
    parser.add_argument("--prediction-field", default="combined_reasoning",
                        help="Generated text to judge (default: issue + arguments + deliberation + explanation).")
    parser.add_argument("--reference-field", default="Simplified_Reasoning")
    parser.add_argument("--key-field", default="document_id")
    parser.add_argument("--pred-label", default="Generated Legal Analysis",
                        help="Name of the judged text inside the judge prompts.")
    parser.add_argument("--ref-label", default="Actual Court Reasoning",
                        help="Name of the reference text inside the judge prompts.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low", choices=("default", "none", "low", "medium", "high"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8, help="Cases judged concurrently.")
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N generated records.")
    parser.add_argument("--max-pred-chars", type=int, default=24000)
    parser.add_argument("--max-label-chars", type=int, default=24000)
    parser.add_argument("--max-facts-chars", type=int, default=12000)
    parser.add_argument("--max-items", type=int, default=10, help="Statutes / precedents shown to the judge.")
    parser.add_argument("--max-item-chars", type=int, default=1500, help="Text excerpt per statute / precedent.")
    parser.add_argument("--no-judgments", dest="include_judgments", action="store_false",
                        help="Show precedent titles only, without judgment excerpts.")
    parser.add_argument("--rejudge", action="store_true", help="Ignore cached judgments.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if is_anthropic(args.model):
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key.startswith("sk-ant-"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing or not an Anthropic key (.env or environment).")
    else:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key.startswith("sk-or-"):
            raise RuntimeError("OPENROUTER_API_KEY is missing or not an OpenRouter key (.env or environment).")

    generated = [r for r in iter_json_array(args.generated, array_key="results")]
    if args.limit is not None:
        generated = generated[: args.limit]
    keys = {record_key(r, args.key_field) for r in generated}
    references = {
        record_key(r, args.key_field): text_value(r, args.reference_field)
        for r in iter_json_array(args.ground_truth)
        if record_key(r, args.key_field) in keys
    }
    contexts = None
    if args.context:
        contexts = {record_key(r, args.key_field): r for r in iter_json_array(args.context)
                    if record_key(r, args.key_field) in keys}

    cases = []
    for record in generated:
        key = record_key(record, args.key_field)
        pred = text_value(record, args.prediction_field)
        if not key or not pred or key not in references:
            continue
        context = contexts.get(key, record) if contexts is not None else record
        cases.append((key, clip(pred, args.max_pred_chars), clip(references[key], args.max_label_chars), context))
    if not cases:
        raise RuntimeError("No generated records with text matched the ground truth.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "judgments.jsonl"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists() and not args.rejudge:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("errors") and row.get("model") == args.model:
                    cache[row["key"]] = row
    todo = [c for c in cases if c[0] not in cache]
    print(f"{len(cases)} cases aligned; {len(cache)} cached; judging {len(todo)} with {args.model}", flush=True)

    judge = Judge(api_key, args)
    if todo:
        check_anthropic_key(judge.client, args.model) if judge.client is not None else check_key(api_key)
    started = time.time()
    with open(cache_path, "a", encoding="utf-8") as sink, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(judge.evaluate, *case) for case in todo]
        for done, future in enumerate(futures, 1):
            try:
                row = future.result()
            except FatalAPIError:
                for pending in futures:
                    pending.cancel()
                raise
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if not row["errors"]:
                cache[row["key"]] = row
            else:
                print(f"  {row['key']}: errors {row['errors']}", flush=True)
            if done % 10 == 0 or done == len(todo):
                print(f"  judged {done}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)

    rows, cumulative = [], []
    totals = {m: 0.0 for m in METRICS}
    counts = {m: 0 for m in METRICS}
    tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    cost = 0.0
    for index, (key, _, _, _) in enumerate(cases, 1):
        row = cache.get(key)
        if row is None:
            continue
        scores = row["scores"]
        for m in METRICS:
            if scores.get(m) is not None:
                totals[m] += scores[m]
                counts[m] += 1
        for usage in row.get("usage", {}).values():
            tokens["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            tokens["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            cost += float(usage.get("cost") or 0.0)
        rows.append({"index": len(rows) + 1, "key": key,
                     **{m: ("" if scores.get(m) is None else scores[m]) for m in METRICS}})
        cumulative.append({"index": len(rows), "key": key,
                           **{m: (totals[m] / counts[m] if counts[m] else math.nan) for m in METRICS}})

    fieldnames = ["index", "key", *METRICS]
    for path, data in ((args.output_dir / "per_example.csv", rows), (args.output_dir / "cumulative.csv", cumulative)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    summary = {
        "generated": str(args.generated),
        "ground_truth": str(args.ground_truth),
        "context": str(args.context) if args.context else "generated records",
        "prediction_field": args.prediction_field,
        "reference_field": args.reference_field,
        "judge_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": None if is_anthropic(args.model) else args.temperature,
        "blinded": True,
        "labels": {"prediction": args.pred_label, "reference": args.ref_label},
        "aligned_cases": len(cases),
        "judged_cases": len(rows),
        "failed_cases": len(cases) - len(rows),
        "metrics": {m: (round(totals[m] / counts[m], 4) if counts[m] else None) for m in METRICS},
        "scored_counts": counts,
        "usage": tokens | ({"cost_usd": round(cost, 4)} if cost else {}),
        "notes": {
            "content_coverage": "core issues/findings of the actual reasoning present in the analysis",
            "consistency": "absence of hallucination / contradiction w.r.t. the actual reasoning",
            "faithfulness": "claims supported by the facts, statutes and precedents given to the model",
            "citation_context_quality": "cited statutes/precedents described accurately (null if none cited)",
            "narrative_quality": "grammar, logical flow, legal tone, readability",
        },
    }
    if not rows:
        raise RuntimeError("No case was judged successfully; summary.json not written.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
