"""Claude Opus 5.5 full run on retrieved (RAG) context, without gold fallbacks.

Same JSON output schema as inference_Claude_Opus_5_5.py. Statutes and
precedents come from data/test_data/test_v2_rag2.json (cases 1-300). Unlike the
original Opus prompt (precedent titles only), retrieved precedents are given
with judgment excerpts, as in the GPT/Nemotron prompts, because a retrieved
title alone carries no information. Missing model outputs are never filled
from ground truth. See rag_inference_common.py.

    python code/inference/inference_Claude_Opus_5_5_rag.py   # -> logs/claude_opus_5_5_full_run.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_inference_common import ModelSpec, main  # noqa: E402


def build_prompt(facts: str, statutes: str, precedents: str) -> str:
    return (
        "You are evaluating an Indian legal appeal.\n"
        "Return a prediction of 1 if the appeal is accepted or 0 if it is rejected.\n"
        "Return exactly one valid JSON object with no markdown. Use this schema:\n"
        '{"legal_issue":"","petitioner_arguments":"","respondent_arguments":""'
        ',"deliberation":"","prediction":0,"explanation":""}\n'
        "Populate every field.\n\n"
        f"Case proceedings:\n{facts}\n\n"
        f"Relevant statutes:\n{statutes}\n\n"
        f"Cited precedents:\n{precedents}\n"
    )


SPEC = ModelSpec(
    model="anthropic/claude-opus-5.5",
    name="claude_opus_5_5",
    title="Claude Opus 5.5",
    build_prompt=build_prompt,
    output_style="json",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
