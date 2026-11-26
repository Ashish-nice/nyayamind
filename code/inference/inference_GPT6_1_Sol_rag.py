"""GPT-6.1-Sol full run on retrieved (RAG) context, without gold fallbacks.

Same prompt as inference_GPT6_1_Sol.py, except that the four analysis sections
are requested as visible headings instead of inside a <think>...</think> block
(GPT-6.1-Sol treats <think> as hidden reasoning and omits it, which is what made
the original script fall back to gold text). Statutes and precedents come from
data/test_data/test_v2_rag2.json (cases 1-300) and missing model outputs are
never filled from ground truth. See rag_inference_common.py.

    python code/inference/inference_GPT6_1_Sol_rag.py            # -> logs/gpt_6_1_sol_full_run.json
    python code/inference/inference_GPT6_1_Sol_rag.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_inference_common import ModelSpec, main  # noqa: E402

SYSTEM_MSG = (
    "You are a smart and intelligent legal assistant for the Indian legal "
    "domain. Based on the user's instructions, assist with tasks related to "
    "the Indian legal system. Since these tasks have legal application, only "
    "provide responses you are extremely certain about, and avoid ambiguity "
    "or uncertainty. Ensure that your output follows the requirements.\n"
)


def build_prompt(facts: str, statutes: str, precedents: str) -> str:
    return (
        "SYSTEM:\n" + SYSTEM_MSG + "\nUSER:\n\n"
        "You are a legal expert tasked with making a judgment about whether an "
        "appeal should be accepted or rejected based on the provided case "
        "proceeding, cited statutes and cited cases. Your task is to evaluate "
        "whether the appeal should be accepted (1) or rejected (0) based on the "
        "input. Use exactly 1 for accepted and 0 for rejected.\n\n"
        "### Now, evaluate the following case:\n"
        f"Case Proceedings: {facts}\n\n"
        f"Relevant Statutes:\n{statutes}\n\n"
        f"Cited Cases Reference:\n{precedents}\n\n"
        "Return exactly this format, with every section populated using the "
        "provided case and legal materials. Write all six sections in your "
        "visible answer (not only in hidden reasoning):\n"
        "**Legal Issue Analysis:**\n[issue and governing legal question]\n\n"
        "**Arguments of Petitioner:**\n[petitioner's facts and legal arguments]\n\n"
        "**Arguments of Respondent:**\n[respondent's facts and legal arguments]\n\n"
        "**Deliberation:**\n[application of statutes and precedents to the facts]\n\n"
        "##PREDICTION: [Insert 0 or 1]\n"
        "##EXPLANATION: [Concise legal reasoning supporting the prediction]\n"
        "Do not leave any section empty, do not use placeholders, and do not "
        "include text outside this format."
    )


SPEC = ModelSpec(
    model="openai/gpt-6.1-sol",
    name="gpt_6_1_sol",
    title="GPT-6.1-Sol",
    build_prompt=build_prompt,
    output_style="think",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
