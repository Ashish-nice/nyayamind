"""Nemotron 3 Ultra 550B A55B full run on retrieved (RAG) context, without gold fallbacks.

Same prompt and output format as inference_Nemotron_3_Ultra_550B_A55B.py, but
statutes and precedents come from data/test_data/test_v2_rag2.json (cases
1-300) and missing model outputs are never filled from ground truth. See
rag_inference_common.py.

    python code/inference/inference_Nemotron_3_Ultra_550B_A55B_rag.py   # -> logs/nemotron_3_ultra_full_run.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_GPT6_1_Sol_rag import build_prompt  # identical prompt  # noqa: E402
from rag_inference_common import ModelSpec, main  # noqa: E402

SPEC = ModelSpec(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    name="nemotron_3_ultra",
    title="NVIDIA Nemotron 3 Ultra 550B A55B",
    build_prompt=build_prompt,
    output_style="think",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
