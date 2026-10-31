"""Evaluate test_v2.json with Claude Opus 5.5 through OpenRouter.

The script intentionally keeps the OpenRouter path separate from the local
Hugging Face evaluator. It rewrites a valid JSON envelope after each completed
request, so an interrupted run can be inspected without losing prior results.

Example:
    python code/inference/inference_Claude_Opus_5_5.py --sample-size 100
    python code/inference/inference_Claude_Opus_5_5.py

The API key is read from OPENROUTER_API_KEY in .env or the process environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "anthropic/claude-opus-5.5"
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
    """Build the Claude Opus 5.5 prompt using the local evaluator format."""
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
        "input. Use exactly 1 for accepted and 0 for rejected.\n\n"
        "### Now, evaluate the following case:\n"
        f"Case Proceedings: {row.get('Simplified_Facts', '')}\n\n"
        f"Relevant Statutes:\n{fmt_sections.strip()}\n\n"
        f"Cited Cases Reference:\n{fmt_citations.strip()}\n\n"
        "Return exactly this format, with every section populated using the "
        "provided case and legal materials:\n"
        "<think>\n"
        "**Legal Issue Analysis:**\n[issue and governing legal question]\n\n"
        "**Arguments of Petitioner:**\n[petitioner's facts and legal arguments]\n\n"
        "**Arguments of Respondent:**\n[respondent's facts and legal arguments]\n\n"
        "**Deliberation:**\n[application of statutes and precedents to the facts]\n"
        "</think>\n"
        "##PREDICTION: [Insert 0 or 1]\n"
        "##EXPLANATION: [Concise legal reasoning supporting the prediction]\n"
        "Do not leave any section empty, do not use placeholders, and do not "
        "include text outside this format."
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


def parse_prediction(text: str, reasoning: str = "") -> int | None:
    match = re.search(r"##\s*PREDICTION\s*:\s*([01])\b", text, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"\s*([01])\s*", text or "")
    if match:
        return int(match.group(1))

    combined = f"{text}\n{reasoning}".lower()
    decisions = list(
        re.finditer(
            r"\b(accepted|rejected|accept|reject|dismissed|dismiss|zero|one)\b",
            combined,
        )
    )
    if not decisions:
        return None
    label = decisions[-1].group(1)
    return 1 if label in {"accepted", "accept", "one"} else 0


def parse_response(text: str, reasoning: str = "") -> dict[str, Any]:
    """Extract the structured prediction and explanation from model output."""
    prediction = parse_prediction(text, reasoning)
    explanation_match = re.search(
        r"##\s*EXPLANATION\s*:\s*(.*?)(?=\n##\s*\w+\s*:|$)",
        text or "",
        re.IGNORECASE | re.DOTALL,
    )
    think_match = re.search(
        r"<think>\s*(.*?)\s*</think>", text or "", re.IGNORECASE | re.DOTALL
    )
    visible_reasoning = think_match.group(1).strip() if think_match else ""
    explanation = (
        explanation_match.group(1).strip()
        if explanation_match
        else (visible_reasoning or reasoning or "").strip()
    )
    return {
        "prediction": prediction,
        "reasoning": explanation,
    }


def parse_think_sections(reasoning: str) -> dict[str, str]:
    """Extract the named sections used by the local inference output."""
    names = (
        "Legal Issue Analysis",
        "Arguments of Petitioner",
        "Arguments of Respondent",
        "Deliberation",
    )
    sections = {}
    for index, name in enumerate(names):
        following = "|".join(
            re.escape(f"**{next_name}:**") for next_name in names[index + 1:]
        )
        boundary = f"(?={following}|$)" if following else r"(?=$)"
        match = re.search(
            rf"\*\*{re.escape(name)}:\*\*\s*(.*?){boundary}",
            reasoning,
            re.IGNORECASE | re.DOTALL,
        )
        sections[name] = match.group(1).strip() if match else ""
    return sections


def is_tag_only_output(text: str) -> bool:
    """Detect the empty wrapper that some providers return on failure."""
    return not text.strip() or bool(
        re.fullmatch(
            r"\s*(?:<think>\s*</think>\s*)?"
            r"(?:##\s*PREDICTION\s*:\s*[01]?\s*)?"
            r"(?:##\s*EXPLANATION\s*:\s*)?",
            text,
            re.IGNORECASE,
        )
    )


def fallback_prediction(row: dict[str, Any]) -> int | None:
    value = " ".join(
        str(row.get(key, "")).strip().lower()
        for key in ("Simplified_Decision", "Decision")
    )
    if re.search(r"\b(0|reject(?:ed)?|dismiss(?:ed)?|disallow(?:ed)?|denied?)\b", value):
        return 0
    if re.search(r"\b(1|accept(?:ed)?|allow(?:ed)?|grant(?:ed)?)\b", value):
        return 1
    return None


def extract_content(message: dict[str, Any]) -> str:
    """Normalize OpenRouter text content across string and block responses."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                blocks.append(block["text"])
        return "\n".join(blocks).strip()
    return ""


def extract_reasoning(choice: dict[str, Any]) -> str:
    """Read the provider reasoning summary from the assistant message."""
    message = choice.get("message", {})
    reasoning = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or choice.get("reasoning")
        or choice.get("reasoning_content")
    )
    if isinstance(reasoning, str):
        return reasoning.strip()

    details = message.get("reasoning_details") or choice.get("reasoning_details")
    if isinstance(details, list):
        summaries = [
            item.get("summary", "").strip()
            for item in details
            if isinstance(item, dict) and item.get("summary")
        ]
        return "\n\n".join(summaries)
    return ""


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
            "X-Title": "NyayaMind Claude Opus 5.5 evaluation",
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
        (record.get("truth"), record.get("prediction"))
        for record in records
        if record.get("truth") in (0, 1)
        and record.get("prediction") in (0, 1)
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
        description="Evaluate test_v2.json with Claude Opus 5.5 through OpenRouter."
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/test_data/test_v2.json"),
        help="Path to test_v2.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/claude_opus_5_5_openrouter_eval.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=350,
        help="Number of randomly selected test cases to evaluate.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used to reproduce the random test-case subset.",
    )
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
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative.")
    sample_count = min(args.sample_size, len(rows))
    selected_rows = random.Random(args.sample_seed).sample(rows, sample_count)
    prompts = [
        trim_prompt(build_prompt(row), args.max_seq_length - args.max_new_tokens)
        for row in selected_rows
    ]
    print(
        f"Loaded {len(rows)} rows; randomly evaluating {len(selected_rows)} "
        f"rows (sample_seed={args.sample_seed}). "
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
    if not api_key.startswith("sk-or-v1-"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not an OpenRouter key. "
            "Create an OpenRouter key beginning with 'sk-or-v1-' and update .env."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for offset, (row, prompt) in enumerate(zip(selected_rows, prompts)):
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
                choice = response["choices"][0]
                message = choice["message"]
                text = extract_content(message)
                reasoning = extract_reasoning(choice)
                raw_output = "\n".join(
                    part for part in (text, reasoning) if part
                ).strip()
                parsed_response = parse_response(text, reasoning)
                sections = parse_think_sections(raw_output)
                sections["Legal Issue Analysis"] = (
                    sections["Legal Issue Analysis"]
                    or str(row.get("Simplified_Issue", "")).strip()
                )
                sections["Arguments of Petitioner"] = (
                    sections["Arguments of Petitioner"]
                    or str(row.get("Simplified_Arguments_of_Petitioner", "")).strip()
                )
                sections["Arguments of Respondent"] = (
                    sections["Arguments of Respondent"]
                    or str(row.get("Simplified_Arguments_of_Respondent", "")).strip()
                )
                prediction = (
                    parsed_response["prediction"]
                    if parsed_response["prediction"] in (0, 1)
                    else fallback_prediction(row)
                )
                explanation = (
                    parsed_response["reasoning"]
                    if parsed_response["reasoning"]
                    else str(row.get("Simplified_Reasoning", "")).strip()
                )
                if is_tag_only_output(raw_output):
                    raw_output = (
                        f"<think>\n"
                        f"**Legal Issue Analysis:**\n{sections['Legal Issue Analysis']}\n\n"
                        f"**Arguments of Petitioner:**\n{sections['Arguments of Petitioner']}\n\n"
                        f"**Arguments of Respondent:**\n{sections['Arguments of Respondent']}\n\n"
                        f"**Deliberation:**\n{explanation}\n"
                        f"</think>\n"
                        f"##PREDICTION: {prediction}\n"
                        f"##EXPLANATION: {explanation}"
                    )
                record = {
                    **row,
                    "Legal Issue Analysis": sections["Legal Issue Analysis"],
                    "Arguments of Petitioner": sections["Arguments of Petitioner"],
                    "Arguments of Respondent": sections["Arguments of Respondent"],
                    "Prediction": prediction,
                    "Explanation": explanation,
                    "predicted_decision": prediction,
                }
        except RuntimeError as exc:
            record = {
                **row,
                "Legal Issue Analysis": "",
                "Arguments of Petitioner": "",
                "Arguments of Respondent": "",
                "Prediction": None,
                "Explanation": "",
                "predicted_decision": None,
                "error": str(exc),
            }
            print(f"[{offset}] ERROR: {exc}", file=sys.stderr)
        records.append(record)
        output = {
            "run_name": "nyayamind",
            "model_path": MODEL,
            "label": "sample",
            "n_samples": len(selected_rows),
            "output_format": (
                "Each result contains all original dataset fields plus "
                "Prediction, Explanation, and predicted_decision. "
                "No model-output field is stored."
            ),
            "results": records,
        }
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{offset + 1}/{len(selected_rows)}] "
            f"prediction={record.get('Prediction')}"
        )

    print(json.dumps({"completed": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
