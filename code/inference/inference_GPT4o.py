"""Evaluate test_v2.json with OpenAI GPT-4o through OpenRouter.

The script intentionally keeps the OpenRouter path separate from the local
Hugging Face evaluator. It writes one JSON record per completed request, so an
interrupted run can be inspected or resumed without losing prior results.

Example:
    python code/inference/inference_GPT4o.py --max-samples 100
    python code/inference/inference_GPT4o.py

The API key is read from OPENROUTER_API_KEY in .env or the process environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "openai/gpt-4o"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_SEQUENCE_LENGTH = 16384
NO_REPEAT_NGRAM_SIZE = 6
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_MAX_RETRIES = 5


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting existing variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_prompt(row: dict[str, Any]) -> str:
    """Build the GPT-4o prompt using the shared case context format."""
    sec_titles = row.get("section_titles", []) or []
    sec_texts = row.get("section_texts", []) or []
    case_titles = row.get("cited_cases", []) or []
    case_judgments = row.get("cited_case_judgments", []) or []

    fmt_sections = (
        "".join(
            f"Statute {index}. {title}:\n   {text}\n\n"
            for index, (title, text) in enumerate(
                zip(sec_titles, sec_texts), 1
            )
        )
        if sec_titles
        else "No specific statutes provided."
    )
    fmt_citations = (
        "".join(
            f"Precedent {index}. {case}:\n"
            f"   {judgment or 'Details not available.'}\n\n"
            for index, (case, judgment) in enumerate(
                zip(case_titles, case_judgments), 1
            )
        )
        if case_titles
        else "No citations provided."
    )

    system_msg = (
        "You are a smart and intelligent legal assistant for the Indian legal "
        "domain. Based on the user's instructions, assist with tasks related to "
        "the Indian legal system. Since these tasks have legal application, only "
        "provide responses you are extremely certain about, and avoid ambiguity "
        "or uncertainty. Ensure that your output follows the requirements.\n"
    )
    return (
        "SYSTEM:\n"
        + system_msg
        + "\nUSER:\n\n"
        "You are a legal expert tasked with making a judgment about whether an "
        "appeal should be accepted or rejected based on the provided case "
        "proceeding, cited statutes and cited cases. Your task is to evaluate "
        "whether the appeal should be accepted (1) or rejected (0) based on the "
        "input.\n\n"
        "### Now, evaluate the following case:\n"
        f"Case Proceedings: {row.get('Simplified_Facts', '')}\n\n"
        f"Relevant Statutes:\n{fmt_sections.strip()}\n\n"
        f"Cited Cases Reference:\n{fmt_citations.strip()}\n\n"
        "Provide your judgment by strictly following this format:\n"
        "##PREDICTION: [Insert 0 or 1]\n"
        "##EXPLANATION: [Insert your reasoning here that led to your prediction.]\n"
        "Strictly do not include anything outside this format. Strictly follow "
        "the provided format. Do not generate placeholders. Just provide the "
        "final judgment and explanation."
    )


def trim_prompt(prompt: str, max_input_tokens: int) -> str:
    """Apply a conservative character-based cap when no local tokenizer exists."""
    # English legal text averages roughly four characters per token. Keeping a
    # small margin prevents an occasional tokenizer outlier from exceeding the
    # requested context budget.
    max_chars = max_input_tokens * 4
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars].rstrip() + "\n\n[Case text truncated.]"


def parse_prediction(text: str) -> int | None:
    match = re.search(r"##\s*PREDICTION\s*:\s*([01])\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def request_completion(
    api_key: str,
    prompt: str,
    timeout: int,
    max_retries: int,
    max_new_tokens: int,
    temperature: float,
    frequency_penalty: float,
    presence_penalty: float,
    repetition_penalty: float,
    seed: int | None,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    if seed is not None:
        payload["seed"] = seed
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ashish-nice/nyayamind",
            "X-Title": "NyayaMind GPT-4o evaluation",
        },
        method="POST",
    )

    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"OpenRouter response has no choices: {body}")
            return body
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == max_retries:
                raise RuntimeError(
                    f"OpenRouter HTTP {exc.code}: {error_body[:1000]}"
                ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        time.sleep(min(60, 2**attempt))

    raise RuntimeError("OpenRouter request failed after retries.")


def decision_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (record["truth"], record["predicted_decision"])
        for record in records
        if record.get("truth") in (0, 1)
        and record.get("predicted_decision") in (0, 1)
    ]
    if not pairs:
        return {"decision_valid_count": 0}

    tp = sum(truth == 1 and pred == 1 for truth, pred in pairs)
    tn = sum(truth == 0 and pred == 0 for truth, pred in pairs)
    fp = sum(truth == 0 and pred == 1 for truth, pred in pairs)
    fn = sum(truth == 1 and pred == 0 for truth, pred in pairs)
    f1_scores = []
    for label in (0, 1):
        true_positive = tp if label == 1 else tn
        false_positive = fp if label == 1 else fn
        false_negative = fn if label == 1 else fp
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            2 * true_positive / denominator if denominator else 0.0
        )
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "decision_accuracy": (tp + tn) / len(pairs),
        "decision_macro_f1": sum(f1_scores) / 2,
        "decision_mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "decision_valid_count": len(pairs),
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate test_v2.json with GPT-4o through OpenRouter."
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/test_data/test_v2.json"),
        help="Path to test_v2.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/gpt4o_openrouter_eval.jsonl"),
        help="JSONL output path",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum prompt plus completion tokens, matching Qwen3 inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens, matching Qwen3 inference.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature; 0.0 keeps decoding deterministic.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=DEFAULT_FREQUENCY_PENALTY,
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=DEFAULT_PRESENCE_PENALTY,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="OpenRouter repetition penalty equivalent to Qwen3's setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional provider seed; omitted by default.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate loading and prompt construction without calling OpenRouter.",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not args.dataset.is_absolute():
        args.dataset = project_root / args.dataset
    if not args.output.is_absolute():
        args.output = project_root / args.output

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {args.dataset}")
    if args.max_seq_length <= args.max_new_tokens:
        raise ValueError("--max-seq-length must exceed --max-new-tokens.")
    end = len(rows) if args.max_samples is None else min(
        len(rows), args.start + args.max_samples
    )
    selected_rows = rows[args.start:end]
    prompts = [
        trim_prompt(build_prompt(row), args.max_seq_length - args.max_new_tokens)
        for row in selected_rows
    ]
    print(
        f"Loaded {len(rows)} rows; evaluating rows {args.start}..{end - 1}. "
        f"Settings: max_new_tokens={args.max_new_tokens}, "
        f"max_seq_length={args.max_seq_length}, "
        f"temperature={args.temperature}, "
        f"frequency_penalty={args.frequency_penalty}, "
        f"presence_penalty={args.presence_penalty}, "
        f"repetition_penalty={args.repetition_penalty}, "
        f"no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE} (local-only), "
        f"seed={args.seed}"
    )
    if args.dry_run:
        print(f"Dry run successful. First prompt length: {len(prompts[0]) if prompts else 0} characters.")
        return 0

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Put the key in .env or set it in the environment."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with args.output.open("a", encoding="utf-8") as output_file:
        for offset, (row, prompt) in enumerate(zip(selected_rows, prompts), args.start):
            started = time.monotonic()
            try:
                response = request_completion(
                    api_key,
                    prompt,
                    args.timeout,
                    args.max_retries,
                    args.max_new_tokens,
                    args.temperature,
                    args.frequency_penalty,
                    args.presence_penalty,
                    args.repetition_penalty,
                    args.seed,
                )
                text = response["choices"][0]["message"].get("content", "").strip()
                usage = response.get("usage", {})
                record = {
                    "index": offset,
                    "truth": int(row["Simplified_Decision"])
                    if str(row.get("Simplified_Decision", "")).strip() in {"0", "1"}
                    else None,
                    "predicted_decision": parse_prediction(text),
                    "raw_output": text,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cost": usage.get("cost"),
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "error": None,
                }
            except RuntimeError as exc:
                record = {
                    "index": offset,
                    "truth": None,
                    "predicted_decision": None,
                    "error": str(exc),
                    "latency_seconds": round(time.monotonic() - started, 3),
                }
                print(f"[{offset}] ERROR: {exc}", file=sys.stderr)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            records.append(record)
            print(
                f"[{offset + 1}/{end}] prediction={record.get('predicted_decision')} "
                f"cost={record.get('cost', 'n/a')} "
                f"time={record['latency_seconds']}s"
            )

    total_cost = sum(record.get("cost") or 0 for record in records)
    print(json.dumps({**decision_metrics(records), "total_cost": total_cost}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
