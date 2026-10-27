"""
eval_base.py  —  Evaluate the BASE model on the test split.

Flow
----
1. PILOT  — sample --pilot_size random rows, run full eval (loss + generation),
             write  <logging_dir>/base_eval_pilot.json,
             print  per-batch timing and ETA for the full dataset.

2. FULL   — run on the entire test set, streaming completed records to
             <logging_dir>/base_eval.json  after every generation batch
             so progress is never lost if the job is interrupted.

Both output files have FIXED names (not tied to any run_name).

Multi-GPU usage:
    CUDA_VISIBLE_DEVICES=1,4,5 python inference_base_model.py \\
        --model_name       unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit \\
        --logging_dir      ./logs \\
        --dataset_dir      ./dataset \\
        --max_seq_length   16384 \\
        --max_new_tokens   4096 \\
        --eval_batch_size  4 \\
        --load_in_4bit \\
        --multi_gpu \\
        2>&1 | tee base_eval_output.txt

Flags
-----
--pilot_size N    Rows to sample for the pilot run (default: 100).
--pilot_seed N    Random seed for reproducible sampling (default: 42).
--pilot_only      Stop after the pilot; skip the full dataset run.
--skip_pilot      Skip the pilot and go straight to the full run.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — works from project root or from inside modules/
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import math
import time
import random
import argparse
import torch
torch._dynamo.config.disable = True
from unsloth import FastLanguageModel

# All shared helpers live in eval.py — single source of truth
from modules.eval import (
    _setup_logger,
    _load_raw_test,
    _build_prompt_text,
    _build_target_text,
    _build_phi4_prompt_text,
    _build_phi4_target_text,
    _build_qwen3_prompt_text,
    _build_qwen3_target_text,
    _tokenize_for_loss,
    _compute_sample_loss,
    _generate_batch,
    _load_model,
    _json_safe,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _get_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the BASE model on the test split. "
            "Runs a fast pilot first (--pilot_size rows), prints ETA, "
            "then runs the full dataset. Results stream to disk after each batch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", type=str,
                   default="unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
                   help="HuggingFace model ID or local path to the base model")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load in 4-bit (NF4) quantisation")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="Load in 8-bit quantisation")
    p.add_argument("--multi_gpu", action="store_true",
                   help="Shard model across all CUDA_VISIBLE_DEVICES via "
                        "device_map='auto'. Default: single GPU (device_map={'':0})")
    p.add_argument("--max_seq_length", type=int, default=8192,
                   help="Max total tokens (prompt + response)")
    p.add_argument("--max_new_tokens", type=int, default=2048,
                   help="Max tokens to generate per sample")
    p.add_argument("--logging_dir", type=str, default="./logs",
                   help="Directory where output JSON files are written")
    p.add_argument("--dataset_dir", type=str, default="./dataset",
                   help="Directory containing test_v2.json")
    p.add_argument("--eval_batch_size", type=int, default=1,
                   help="Batch size for generation. Loss pass is always 1. "
                        "Try 2-4 for a 14B 4-bit model if VRAM allows.")
    p.add_argument("--heartbeat_tokens", type=int, default=50,
                   help="Log a heartbeat every N generated tokens per sample "
                        "so you can confirm the model is still running.")
    p.add_argument("--pilot_size", type=int, default=100,
                   help="Number of randomly sampled rows for the pilot run")
    p.add_argument("--pilot_seed", type=int, default=42,
                   help="Random seed for reproducible pilot sampling")
    p.add_argument("--pilot_only", action="store_true",
                   help="Stop after the pilot run; skip the full dataset")
    p.add_argument("--skip_pilot", action="store_true",
                   help="Skip the pilot and go straight to the full run")
    p.add_argument("--full_precision", action="store_true",
               help="Load model in bf16, no quantization")
    p.add_argument("--train_phi4", action="store_true",
               help="Evaluate Phi-4 reasoning models instead of DeepSeek")
    p.add_argument("--train_qwen3", action="store_true",
               help="Evaluate Qwen 3.5 reasoning models instead of DeepSeek")

    args = p.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        p.error("--load_in_4bit and --load_in_8bit are mutually exclusive.")
    if not args.load_in_4bit and not args.load_in_8bit and not args.full_precision:
        args.load_in_4bit = True  # default only when no precision flag given
    if args.pilot_only and args.skip_pilot:
        p.error("--pilot_only and --skip_pilot are mutually exclusive.")
    return args


# ---------------------------------------------------------------------------
# Streaming JSON writer
# ---------------------------------------------------------------------------

class _StreamingWriter:
    """
    Writes a JSON file incrementally.

    Structure:
        {
          <header fields>,
          "results": [
            { ...record... },
            { ...record... },
            ...
          ],
          <footer fields appended on close()>
        }

    Call write_batch() after each generation batch.
    Call close()       when all batches are done.
    """

    def __init__(self, path: str, header: dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path  = path
        self._first = True
        with open(path, "w", encoding="utf-8") as f:
            f.write("{\n")
            for k, v in header.items():
                f.write(f"  {json.dumps(k)}: {json.dumps(v)},\n")
            f.write('  "results": [\n')

    def write_batch(self, records: list):
        with open(self._path, "a", encoding="utf-8") as f:
            for rec in records:
                if not self._first:
                    f.write(",\n")
                f.write("    " + json.dumps(_json_safe(rec), ensure_ascii=False))
                self._first = False

    def close(self, footer: dict = None):
        with open(self._path, "a", encoding="utf-8") as f:
            f.write("\n  ]")
            if footer:
                for k, v in footer.items():
                    f.write(f",\n  {json.dumps(k)}: {json.dumps(v)}")
            f.write("\n}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    """Format a duration as  Xh Ym Zs / Ym Zs / Zs."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:   return f"{h}h {m}m {s}s"
    if m:   return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

def _eval_loop(model, tokenizer, rows, args, device, logger,
               out_path, label, n_full=None):
    """
    Run loss pass + streaming generation pass over `rows`.

    Parameters
    ----------
    label  : "PILOT" or "FULL" (used in log messages)
    n_full : total size of the full dataset — when set, ETA for the full run
             is shown during the pilot generation pass.

    Returns
    -------
    (agg_loss, agg_ppl, gen_elapsed_seconds)
    """
    n = len(rows)
    w = len(str(n))

    # Select prompt builders based on model type
    use_phi4 = getattr(args, 'train_phi4', False)
    use_qwen3 = getattr(args, 'train_qwen3', False)
    if use_phi4:
        build_prompt = _build_phi4_prompt_text
        build_target = _build_phi4_target_text
        gen_suffix = ""  # Already in the ChatML template
    elif use_qwen3:
        build_prompt = _build_qwen3_prompt_text
        build_target = _build_qwen3_target_text
        gen_suffix = ""  # Already in the ChatML template
    else:
        build_prompt = _build_prompt_text
        build_target = _build_target_text
        gen_suffix = "ASSISTANT:\n"

    # ── Pass 1: loss ──────────────────────────────────────────────────────────
    logger.info(f"[{label}] Pass 1/2 — loss pass  ({n} rows, batch_size=1) …")
    losses     = []
    t0         = time.time()

    for i, row in enumerate(rows):
        batch = _tokenize_for_loss(
            build_prompt(row), build_target(row),
            tokenizer, args.max_seq_length, args.max_new_tokens,
        )
        losses.append(_compute_sample_loss(model, batch, device))

        if (i + 1) % 25 == 0 or (i + 1) == n:
            valid   = [l for l in losses if not math.isnan(l)]
            avg     = sum(valid) / len(valid) if valid else float("nan")
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (n - (i + 1)) / rate if rate > 0 else float("nan")
            logger.info(
                f"  [{label}] loss [{i+1:{w}}/{n}]  "
                f"avg: {avg:.4f}  elapsed: {_fmt(elapsed)}  ETA: {_fmt(eta)}"
            )

    valid_l  = [l for l in losses if not math.isnan(l)]
    agg_loss = sum(valid_l) / len(valid_l) if valid_l else float("nan")
    agg_ppl  = math.exp(agg_loss) if math.isfinite(agg_loss) else float("nan")
    logger.info(f"[{label}] Loss done.  Aggregate: {agg_loss:.4f}  PPL: {agg_ppl:.2f}")

    # ── Pass 2: generation (batched, streaming to disk) ───────────────────────
    logger.info(
        f"[{label}] Pass 2/2 — generation ({n} rows, sample-by-sample, heartbeat every {getattr(args, 'heartbeat_tokens', 50)} tokens) …"
    )
    prompts = [build_prompt(r) + gen_suffix for r in rows]

    writer = _StreamingWriter(out_path, header={
        "model_path"     : args.model_name,
        "model_type"     : "base",
        "label"          : label.lower(),
        "n_samples"      : n,
        "eval_batch_size": args.eval_batch_size,
    })

    t0        = time.time()
    completed = 0

    for start in range(0, n, args.eval_batch_size):
        end     = min(start + args.eval_batch_size, n)
        outputs = _generate_batch(
            model, tokenizer, prompts[start:end], args, device,
            logger=logger, sample_offset=start
        )

        batch_records = []
        for j, (row, text) in enumerate(zip(rows[start:end], outputs)):
            rec = {k: _json_safe(v) for k, v in row.items()}
            rec["base_model_output"] = text
            rec["per_sample_loss"]   = (
                losses[start + j] if math.isfinite(losses[start + j]) else None
            )
            batch_records.append(rec)

        writer.write_batch(batch_records)    # ← flush to disk immediately
        completed += len(batch_records)

        elapsed = time.time() - t0
        rate    = completed / elapsed if elapsed > 0 else 0
        eta_run = (n - completed) / rate if rate > 0 else float("nan")

        eta_full_str = ""
        if n_full is not None and rate > 0:
            eta_full_str = f"  |  ETA full ({n_full} rows): {_fmt(n_full / rate)}"

        logger.info(
            f"  [{label}] batch done [{completed:{w}}/{n}]  "
            f"{rate:.2f} rows/s  elapsed: {_fmt(elapsed)}  "
            f"ETA this run: {_fmt(eta_run)}{eta_full_str}"
        )

    gen_elapsed = time.time() - t0
    writer.close(footer={
        "aggregate_loss"    : agg_loss if math.isfinite(agg_loss) else None,
        "aggregate_ppl"     : agg_ppl  if math.isfinite(agg_ppl)  else None,
        "total_gen_seconds" : round(gen_elapsed, 1),
    })

    logger.info(f"[{label}] Saved → {out_path}")
    return agg_loss, agg_ppl, gen_elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_base_eval(args, logger):
    logger.info("=" * 60)
    logger.info("BASE MODEL EVAL  —  starting")
    logger.info("=" * 60)
    logger.info(f"model           : {args.model_name}")
    logger.info(f"eval_batch_size : {args.eval_batch_size}")
    logger.info(f"pilot_size      : {args.pilot_size}  seed={args.pilot_seed}")

    # Load model once — reused for both pilot and full run
    model, tokenizer = _load_model(args.model_name, args, logger)
    device = next(model.parameters()).device
    model.eval()
    FastLanguageModel.for_inference(model)

    # Load full test set
    all_rows = list(_load_raw_test(args.dataset_dir))
    n_total  = len(all_rows)
    logger.info(f"Full test size  : {n_total} rows")
    os.makedirs(args.logging_dir, exist_ok=True)

    pilot_path = os.path.join(args.logging_dir, "base_eval_pilot.json")
    full_path  = os.path.join(args.logging_dir, "base_eval.json")

    # ═══════════════════════════════════════════════════════════════
    # PILOT RUN
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_pilot:
        pilot_n       = min(args.pilot_size, n_total)
        rng           = random.Random(args.pilot_seed)
        pilot_indices = sorted(rng.sample(range(n_total), pilot_n))
        pilot_rows    = [all_rows[i] for i in pilot_indices]

        logger.info("")
        logger.info("━" * 60)
        logger.info(f"PILOT RUN — {pilot_n} random rows  (seed={args.pilot_seed})")
        logger.info("━" * 60)

        _, _, pilot_gen_t = _eval_loop(
            model, tokenizer, pilot_rows, args, device, logger,
            out_path=pilot_path,
            label="PILOT",
            n_full=n_total,       # enables ETA-for-full-run in log
        )

        rate = pilot_n / pilot_gen_t if pilot_gen_t > 0 else 0

        logger.info("")
        logger.info("┌─ PILOT SUMMARY " + "─" * 43)
        logger.info(f"│  Rows evaluated  : {pilot_n}")
        logger.info(f"│  Gen time        : {_fmt(pilot_gen_t)}")
        logger.info(f"│  Throughput      : {rate:.2f} rows/s")
        logger.info(f"│  ETA — full run  : {_fmt(n_total / rate if rate > 0 else float('nan'))}"
                    f"  ({n_total} rows @ {rate:.2f} rows/s)")
        logger.info(f"│  Output file     : {pilot_path}")
        logger.info("└" + "─" * 58)

        if args.pilot_only:
            logger.info("--pilot_only set. Stopping here.")
            del model
            torch.cuda.empty_cache()
            return pilot_path, None

    # ═══════════════════════════════════════════════════════════════
    # FULL RUN
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("━" * 60)
    logger.info(f"FULL RUN — {n_total} rows")
    logger.info("━" * 60)

    _eval_loop(
        model, tokenizer, all_rows, args, device, logger,
        out_path=full_path,
        label="FULL",
    )

    logger.info("")
    logger.info("All done.")
    if not args.skip_pilot:
        logger.info(f"  Pilot → {pilot_path}")
    logger.info(f"  Full  → {full_path}")

    del model
    torch.cuda.empty_cache()
    return pilot_path if not args.skip_pilot else None, full_path


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args   = _get_args()
    _logger = _setup_logger("eval_base")
    _logger.info(
        f"CUDA_VISIBLE_DEVICES = "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set (all GPUs visible)')}"
    )
    _logger.info(f"Visible GPU count    = {torch.cuda.device_count()}")
    run_base_eval(_args, _logger)