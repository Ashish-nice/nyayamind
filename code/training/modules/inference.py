"""
eval.py  —  Evaluate the FINE-TUNED model on the test split.

Flow
----
1. PILOT   — sample --pilot_size random rows, run full eval (loss + generation),
             write  <logging_dir>/finetuned_eval_pilot.json,
             print  per-batch timing and ETA for the full dataset.

2. FULL    — run on the entire test set, streaming completed records to
             <logging_dir>/finetuned_eval.json  after every generation batch
             so progress is never lost if the job is interrupted.
"""

import sys
import os

# Must be set BEFORE importing torch — fully disables torch.compile/dynamo.
# Required for Phi-4 whose LongRoPE has dynamic control flow (`if seq_len > max_pos`)
# that torch._dynamo cannot trace.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import math
import time
import random
import argparse
import torch
import threading
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _setup_logger(name: str):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


def _load_raw_test(dataset_dir: str):
    from datasets import load_dataset
    path = os.path.join(dataset_dir, "test_v2.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw test file not found: {path}")
    return load_dataset("json", data_files={"test": path}, split="test")


def show_prompt_output_sample(prompt, generated_text, tokenizer):
    print("\n" + "="*80)
    print("SAMPLE INPUT PROMPT")
    print("="*80)
    print(prompt)

    print("\n" + "="*80)
    print("MODEL OUTPUT")
    print("="*80)
    print(generated_text)

    print("="*80 + "\n")


def _build_prompt_text(row: dict) -> str:
    sec_titles  = row.get("section_titles",       []) or []
    sec_texts   = row.get("section_texts",        []) or []
    case_titles = row.get("cited_cases",          []) or []
    case_judgs  = row.get("cited_case_judgments", []) or []

    fmt_sections = (
        "".join(
            f"Statute {i}. {t}:\n   {x}\n\n"
            for i, (t, x) in enumerate(zip(sec_titles, sec_texts), 1)
        ) if sec_titles else "No specific statutes provided."
    )
    fmt_citations = (
        "".join(
            f"Precedent {i}. {c}:\n   {info or 'Details not available.'}\n\n"
            for i, (c, info) in enumerate(zip(case_titles, case_judgs), 1)
        ) if case_titles else "No citations provided."
    )
    system_msg = (
        "SYSTEM:\nYou are a smart and intelligent legal assistant for the Indian "
        "legal domain. Based on the user's instructions, you will have to perform "
        "or assist in some tasks related to the Indian legal system. Since these "
        "tasks have some legal application, only provide responses you are extremely "
        "certain about, and avoid being ambiguous or uncertain. Ensure that your "
        "outputs adhere to the user's instructions or requirements.\n"
    )
    user_content = (
        "USER:\n\n"
        "    You are a legal expert tasked with making a judgment about whether an "
        "appeal should be accepted or rejected based on the provided case proceeding."
        "Your task is to evaluate whether the appeal "
        "should be accepted (1) or rejected (0) based on the input.\n\n"
        f"    ### Now, evaluate the following case:\n"
        f"    Case Proceedings: {row.get('Simplified_Facts', '')}\n\n"
        "    Provide your judgment by strictly following this format:\n"
        "    ##PREDICTION: [Insert your prediction here]\n"
        "    ##EXPLANATION: [Insert your reasoning here that led you to your prediction.]\n"
        "    Strictly do not include anything outside this format. Strictly follow the "
        "provided format. Do not generate placeholders. Just provide the final "
        "judgment and explanation.\n"
    )
    return system_msg + user_content


def _build_target_text(row: dict) -> str:
    thought = (
        f"**Legal Issue Analysis:**\n"
        f"{row.get('Simplified_Issue', 'Issue analysis not provided.')}\n\n"
        f"**Arguments of Petitioner:**\n"
        f"{row.get('Simplified_Arguments_of_Petitioner', 'Petitioner arguments not provided.')}\n\n"
        f"**Arguments of Respondent:**\n"
        f"{row.get('Simplified_Arguments_of_Respondent', 'Respondent arguments not provided.')}\n\n"
        "**Deliberation:**\n"
        "Weighing the arguments against the relevant statutes and cited cases to form a decision."
    )
    response = (
        f"##PREDICTION: {row.get('Simplified_Decision', '0')}\n"
        f"##EXPLANATION: {row.get('Simplified_Reasoning', 'Reasoning not provided.')}"
    )
    return f"ASSISTANT:\n### Response:\n<think>\n{thought}\n</think>\n{response}"


def _build_phi4_prompt_text(row: dict, use_mini_format: bool = False) -> str:
    """Build Phi-4-reasoning prompt text for evaluation."""
    sec_titles  = row.get("section_titles",       []) or []
    sec_texts   = row.get("section_texts",        []) or []
    case_titles = row.get("cited_cases",          []) or []
    case_judgs  = row.get("cited_case_judgments", []) or []

    fmt_sections = (
        "".join(
            f"Statute {i}. {t}:\n   {x}\n\n"
            for i, (t, x) in enumerate(zip(sec_titles, sec_texts), 1)
        ) if sec_titles else "No specific statutes provided."
    )
    fmt_citations = (
        "".join(
            f"Precedent {i}. {c}:\n   {info or 'Details not available.'}\n\n"
            for i, (c, info) in enumerate(zip(case_titles, case_judgs), 1)
        ) if case_titles else "No citations provided."
    )

    system_prompt = (
        "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. You are a smart and intelligent legal assistant for the Indian legal domain. Based on the user's instructions, you will have to perform or assist in some tasks related to the Indian legal system. Since these tasks have some legal application, only provide responses you are extremely certain about, and avoid being ambiguous or uncertain. Ensure that your outputs adhere to the user's instructions or requirements. Now, try to solve the following question through the above guidelines:"
    )

    user_content = (
        "You are a legal expert tasked with making a judgment about whether an "
        "appeal should be accepted or rejected based on the provided case proceeding, "
        "cited statutes and cited cases. Your task is to evaluate whether the appeal "
        "should be accepted (1) or rejected (0) based on the input.\n\n"
        f"### Now, evaluate the following case:\n"
        f"Case Proceedings: {row.get('Simplified_Facts', '')}\n\n"
        f"Relevant Statutes:\n{fmt_sections.strip()}\n\n"
        f"Cited Cases Reference:\n{fmt_citations.strip()}\n\n"
        "Provide your judgment by strictly following this format:\n"
        "##PREDICTION: [Insert your prediction here]\n"
        "##EXPLANATION: [Insert your reasoning here that led you to your prediction.]\n"
        "Strictly do not include anything outside this format. Strictly follow the "
        "provided format. Do not generate placeholders. Just provide the final "
        "judgment and explanation."
    )

    if use_mini_format:
        return (
            f"<|system|>{system_prompt}<|end|>"
            f"<|user|>{user_content}<|end|>"
            f"<|assistant|>\n"
        )
    else:
        return (
            f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user<|im_sep|>\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant<|im_sep|>\n"
        )


def _build_phi4_target_text(row: dict) -> str:
    """Build Phi-4-reasoning target text for evaluation."""
    thought = (
        f"**Legal Issue Analysis:**\n"
        f"{row.get('Simplified_Issue', 'Issue analysis not provided.')}\n\n"
        f"**Arguments of Petitioner:**\n"
        f"{row.get('Simplified_Arguments_of_Petitioner', 'Petitioner arguments not provided.')}\n\n"
        f"**Arguments of Respondent:**\n"
        f"{row.get('Simplified_Arguments_of_Respondent', 'Respondent arguments not provided.')}\n\n"
        "**Deliberation:**\n"
        "Weighing the arguments against the relevant statutes and cited cases to form a decision."
    )
    response = (
        f"##PREDICTION: {row.get('Simplified_Decision', '0')}\n"
        f"##EXPLANATION: {row.get('Simplified_Reasoning', 'Reasoning not provided.')}"
    )
    return f"<think>\n{thought}\n</think>\n{response}"



def _build_qwen3_prompt_text(row: dict) -> str:
    """Build Qwen 3.5 ChatML prompt text for evaluation."""
    sec_titles  = row.get("section_titles",       []) or []
    sec_texts   = row.get("section_texts",        []) or []
    case_titles = row.get("cited_cases",          []) or []
    case_judgs  = row.get("cited_case_judgments", []) or []

    fmt_sections = (
        "".join(
            f"Statute {i}. {t}:\n   {x}\n\n"
            for i, (t, x) in enumerate(zip(sec_titles, sec_texts), 1)
        ) if sec_titles else "No specific statutes provided."
    )
    fmt_citations = (
        "".join(
            f"Precedent {i}. {c}:\n   {info or 'Details not available.'}\n\n"
            for i, (c, info) in enumerate(zip(case_titles, case_judgs), 1)
        ) if case_titles else "No citations provided."
    )

    system_prompt = (
        "You are a smart and intelligent legal assistant for the Indian legal domain. Based on the user's instructions, you will have to perform or assist in some tasks related to the Indian legal system. Since these tasks have some legal application, only provide responses you are extremely certain about, and avoid being ambiguous or uncertain. Ensure that your outputs adhere to the user's instructions or requirements.\n"
    )

    user_content = (
        "You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided case proceeding. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) based on the input.\n\n"
        f"### Now, evaluate the following case:\n"
        f"Case Proceedings: {row.get('Simplified_Facts', '')}\n\n"
        "Provide your judgment by strictly following this format:\n"
        "##PREDICTION: [Insert your prediction here]\n"
        "##EXPLANATION: [Insert your reasoning here that led you to your prediction.]\n"
        "Strictly do not include anything outside this format. Strictly follow the provided format. Do not generate placeholders. Just provide the final judgment and explanation."
    )

    # Qwen 3.5 ChatML: no <|im_sep|>, uses newline after role
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_qwen3_target_text(row: dict) -> str:
    """Build Qwen 3.5 target text for evaluation."""
    thought = (
        f"**Legal Issue Analysis:**\n"
        f"{row.get('Simplified_Issue', 'Issue analysis not provided.')}\n\n"
        f"**Arguments of Petitioner:**\n"
        f"{row.get('Simplified_Arguments_of_Petitioner', 'Petitioner arguments not provided.')}\n\n"
        f"**Arguments of Respondent:**\n"
        f"{row.get('Simplified_Arguments_of_Respondent', 'Respondent arguments not provided.')}\n\n"
        "**Deliberation:**\n"
        "Weighing the arguments against the relevant statutes and cited cases to form a decision."
    )
    response = (
        f"##PREDICTION: {row.get('Simplified_Decision', '0')}\n"
        f"##EXPLANATION: {row.get('Simplified_Reasoning', 'Reasoning not provided.')}"
    )
    return f"<think>\n{thought}\n</think>\n{response}"


def _tokenize_for_loss(prompt_text, target_text, tokenizer,
                        max_seq_length, max_new_tokens) -> dict:
    target_ids = tokenizer(
        target_text + tokenizer.eos_token,
        add_special_tokens=False, truncation=True,
        max_length=max_new_tokens, return_tensors="pt",
    )["input_ids"][0]

    input_ids = tokenizer(
        prompt_text,
        add_special_tokens=False, truncation=True,
        max_length=max(1, max_seq_length - len(target_ids)),
        return_tensors="pt",
    )["input_ids"][0]

    full_ids = torch.cat([input_ids, target_ids])
    return {
        "input_ids":      full_ids.unsqueeze(0),
        "attention_mask": torch.ones_like(full_ids).unsqueeze(0),
        "labels":         torch.cat([
            torch.full((len(input_ids),), -100, dtype=torch.long),
            target_ids,
        ]).unsqueeze(0),
    }


@torch.no_grad()
def _compute_sample_loss(model, batch, device) -> float:
    if (batch["labels"] != -100).sum().item() == 0:
        return float("nan")
    out = model(
        input_ids      = batch["input_ids"].to(device),
        attention_mask = batch["attention_mask"].to(device),
        labels         = batch["labels"].to(device),
    )
    return out.loss.item()


def _generate_batch(model, tokenizer, prompts, args, device, logger=None,
                    sample_offset=0):
    max_prompt_len  = args.max_seq_length - args.max_new_tokens
    heartbeat_every = getattr(args, "heartbeat_tokens", 50)
    results         = []

    for local_idx, prompt in enumerate(prompts):
        global_idx = sample_offset + local_idx

        enc = tokenizer(
            [prompt], return_tensors="pt",
            truncation=True, max_length=max_prompt_len,
        ).to(device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **enc,
            max_new_tokens    = args.max_new_tokens,
            use_cache         = True,
            do_sample         = False,
            repetition_penalty= 1.1,
            no_repeat_ngram_size=6,
            pad_token_id      = tokenizer.pad_token_id,
            eos_token_id      = tokenizer.eos_token_id,
            streamer          = streamer,
        )

        thread = threading.Thread(
            target=model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        generated_text = ""
        token_count    = 0
        t_start        = time.time()

        for new_text in streamer:
            generated_text += new_text
            token_count += len(new_text.split())
            if logger and token_count % heartbeat_every < len(new_text.split()):
                elapsed = time.time() - t_start
                logger.info(
                    f"    sample {global_idx} : "
                    f"~{token_count} tokens  ({elapsed:.0f}s elapsed)"
                )

        thread.join()
        elapsed = time.time() - t_start
        if logger:
            logger.info(
                f"    sample {global_idx} done : "
                f"~{token_count} tokens  ({elapsed:.0f}s)"
            )

        results.append(generated_text.strip())

    return results


# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
#     if args.multi_gpu:
#         logger.info(f"Multi-GPU : device_map='auto' across "
#                     f"{torch.cuda.device_count()} visible GPU(s)")
#     else:
#         logger.info("Single-GPU: device_map={'': 0}")

#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name    = model_path,
#         max_seq_length= args.max_seq_length,
#         load_in_4bit  = args.load_in_4bit,
#         load_in_8bit  = getattr(args, "load_in_8bit", False),
#         dtype         = None,
#         device_map    = device_map,
#     )
#     return model, tokenizer


# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}

#     if args.multi_gpu:
#         logger.info(f"Multi-GPU : device_map='auto' across "
#                     f"{torch.cuda.device_count()} visible GPU(s)")
#     else:
#         logger.info("Single-GPU: device_map={'': 0}")
    
#     # 8-bit requires the base model to be identified to map target modules
#     load_in_8bit = getattr(args, "load_in_8bit", False)
    
#     if load_in_8bit:
#         logger.info("Detected 8-bit request: Loading base model with adapter mapping...")
#         # In 8-bit, we pass the local checkpoint path. 
#         # Unsloth reads adapter_config.json to find the base model and 
#         # then injects the adapters into the quantized base linear layers.
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name    = model_path, # Path to your 'final_model' folder
#             max_seq_length= args.max_seq_length,
#             load_in_4bit  = False,
#             load_in_8bit  = True,
#             device_map  = device_map,
#         )
#     else:
#         # Standard loading for 4-bit or full precision
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name    = model_path,
#             max_seq_length= args.max_seq_length,
#             load_in_4bit  = args.load_in_4bit,
#             load_in_8bit  = False,
#             dtype         = None,
#             device_map    = device_map,
#         )

#     FastLanguageModel.for_inference(model)
#     return model, tokenizer


# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
#     load_in_8bit = getattr(args, "load_in_8bit", False)

#     if load_in_8bit:
#         logger.info("8-bit Mode: Explicitly loading base model then applying adapters.")
#         # 1. Load the BASE model name (e.g., the original DeepSeek or Qwen model)
#         # We find this name from the adapter_config.json inside your model_path
#         import json
#         with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
#             config = json.load(f)
#             base_model_name = config.get("base_model_name_or_path")

#         # 2. Initialize the base model in 8-bit
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = base_model_name,
#             max_seq_length = args.max_seq_length,
#             load_in_4bit = False,
#             load_in_8bit = True,
#             device_map = device_map,
#         )

#         # 3. Manually load the adapters from your checkpoint
#         logger.info(f"Applying LoRA adapters from: {model_path}")
#         model.load_adapter(model_path)
        
#     else:
#         # Standard flow for 4-bit (which Unsloth handles natively without mapping errors)
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = model_path,
#             max_seq_length = args.max_seq_length,
#             load_in_4bit = args.load_in_4bit,
#             load_in_8bit = False,
#             device_map = device_map,
#         )

#     FastLanguageModel.for_inference(model)
#     return model, tokenizer


# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
    
#     # Check if we are specifically requesting 8-bit
#     load_in_8bit = getattr(args, "load_in_8bit", False)

#     if load_in_8bit:
#         logger.info("8-bit Quantization detected: Applying sequential loading fix.")
        
#         # 1. Manually retrieve the base model name from your adapter's config
#         import json
#         config_path = os.path.join(model_path, "adapter_config.json")
#         if not os.path.exists(config_path):
#             raise FileNotFoundError(f"Missing adapter_config.json in {model_path}")
            
#         with open(config_path, "r") as f:
#             config = json.load(f)
#             # This ensures we load the exact base model used during training
#             base_model_name = config.get("base_model_name_or_path")

#         logger.info(f"Step 1: Loading Base Model in 8-bit: {base_model_name}")
        
#         # 2. Load the BASE model in 8-bit first
#         # This initializes the linear layers (q_proj, v_proj, etc.) in 8-bit state
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name    = base_model_name,
#             max_seq_length= args.max_seq_length,
#             load_in_4bit  = False,
#             load_in_8bit  = True,
#             device_map    = device_map,
#         )

#         # 3. Explicitly attach the fine-tuned adapters to the already-quantized base
#         logger.info(f"Step 2: Attaching adapters from: {model_path}")
#         from peft import PeftModel
#         model = PeftModel.from_pretrained(model, model_path)
        
#     else:
#         # Standard Unsloth logic for 4-bit (handles native bnb-4bit mapping)
#         logger.info("4-bit or Standard Quantization detected: Using native Unsloth loading.")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name    = model_path,
#             max_seq_length= args.max_seq_length,
#             load_in_4bit  = args.load_in_4bit,
#             load_in_8bit  = False,
#             device_map    = device_map,
#         )

#     # Enable Unsloth's optimized inference kernels
#     FastLanguageModel.for_inference(model)
#     return model, tokenizer

# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
#     load_in_8bit = getattr(args, "load_in_8bit", False)

#     if load_in_8bit:
#         logger.info("Applying 8-bit Atomic Load: Loading base with manual adapter patching.")
        
#         # 1. Get the base model name from the adapter config
#         import json
#         with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
#             base_model_name = json.load(f).get("base_model_name_or_path")

#         # 2. Load the base model in 8-bit first
#         # This ensures the H200 GPUs (which you are using) initialize the weights correctly
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = base_model_name,
#             max_seq_length = args.max_seq_length,
#             load_in_4bit = False,
#             load_in_8bit = True,
#             device_map = device_map,
#         )

#         # 3. Use Unsloth's native method to load the Peft Model
#         # This is the 'secret sauce' that matches the training step logic
#         from peft import PeftModel
#         logger.info(f"Loading PeftModel adapters from {model_path}...")
#         model = PeftModel.from_pretrained(model, model_path)
        
#     else:
#         # Standard logic for 4-bit (Native Unsloth support)
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = model_path,
#             max_seq_length = args.max_seq_length,
#             load_in_4bit = args.load_in_4bit,
#             load_in_8bit = False,
#             device_map = device_map,
#         )

#     FastLanguageModel.for_inference(model)
#     return model, tokenizer


# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
    
#     # Identify if we are specifically requesting 8-bit
#     load_in_8bit = getattr(args, "load_in_8bit", False)

#     if load_in_8bit:
#         logger.info("8-bit Quantization Mode: Loading base then applying adapters.")
        
#         # 1. Use the official DeepSeek base name
#         # We bypass the adapter_config.json to avoid the 'unsloth' path issue
#         base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

#         # 2. Load the base model with trust_remote_code
#         # This ensures the Qwen2-specific linear layers are initialized
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name     = base_model_name,
#             max_seq_length = args.max_seq_length,
#             load_in_4bit   = False,
#             load_in_8bit   = True,
#             device_map     = device_map,
#             trust_remote_code = True, 
#         )

#         # 3. Use the load_adapter method directly
#         # This is Unsloth's optimized way to re-attach the layers
#         logger.info(f"Merging fine-tuned adapters from: {model_path}")
#         model.load_adapter(model_path)
        
#     else:
#         # Standard 4-bit loading logic
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name    = model_path,
#             max_seq_length= args.max_seq_length,
#             load_in_4bit  = args.load_in_4bit,
#             load_in_8bit  = False,
#             device_map    = device_map,
#         )

#     FastLanguageModel.for_inference(model)
#     return model, tokenizer


# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
    
#     # 1. Force the correct base model name for 8-bit
#     # We use the official repo to ensure the architecture is standard
#     base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
#     logger.info(f"Loading Base Model in 8-bit: {base_model_name}")
    
#     # 2. Load the base model with Unsloth's 8-bit optimizations
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = base_model_name,
#         max_seq_length = args.max_seq_length,
#         load_in_4bit = False,
#         load_in_8bit = True,
#         device_map = device_map,
#         trust_remote_code = True,
#     )

#     # 3. Manually attach the adapter using the PeftModel wrapper
#     # This bypasses the automated 'from_pretrained' path that was failing
#     from peft import PeftModel
#     logger.info(f"Manually attaching adapters from: {model_path}")
    
#     try:
#         model = PeftModel.from_pretrained(model, model_path)
#     except ValueError as e:
#         logger.error("Standard injection failed. Attempting forced module mapping...")
#         # If it still fails, it's because the 8-bit layers aren't being seen as linear.
#         # We can try to force the PeftModel to recognize the 8-bit layers.
#         model = PeftModel.from_pretrained(model, model_path, is_trainable=False)

#     # 4. Prepare for inference
#     FastLanguageModel.for_inference(model)
#     return model, tokenizer

# from transformers import BitsAndBytesConfig

# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
#     load_in_8bit = getattr(args, "load_in_8bit", False)

#     # 1. Official Base Model Reference
#     base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

#     if load_in_8bit:
#         logger.info(f"Forcing 8-bit Quantization via BitsAndBytesConfig for: {base_model_name}")
        
#         # Manually define 8-bit config to bypass Unsloth internal mapping issues
#         bnb_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#             llm_int8_threshold=6.0,
#             llm_int8_enable_fp32_cpu_offload=True,
#         )

#         # Load with manual quantization_config
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = base_model_name,
#             max_seq_length = args.max_seq_length,
#             quantization_config = bnb_config, # [CRITICAL FIX]
#             device_map = device_map,
#             trust_remote_code = True,
#         )

#         logger.info(f"Attaching adapters from: {model_path}")
#         from peft import PeftModel
#         model = PeftModel.from_pretrained(model, model_path)
        
#     else:
#         # Standard 4-bit/Full Precision logic
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = model_path,
#             max_seq_length = args.max_seq_length,
#             load_in_4bit = args.load_in_4bit,
#             load_in_8bit = False,
#             device_map = device_map,
#         )

#     FastLanguageModel.for_inference(model)
#     return model, tokenizer

def _load_model(model_path, args, logger):
    device_map = "auto" if args.multi_gpu else {"": 0}
    if args.multi_gpu:
        logger.info(f"Multi-GPU : device_map='auto' across "
                    f"{torch.cuda.device_count()} visible GPU(s)")
    else:
        logger.info("Single-GPU: device_map={'': 0}")

    load_in_8bit = getattr(args, "load_in_8bit", False)

    if load_in_8bit:
        adapter_config_path = os.path.join(model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            # ----------------------------------------------------------------
            # 8-bit Fine-tuned path: Load base model + manually apply LoRA deltas
            # Unsloth cannot inject LoRA adapters into 8-bit layers due to
            # internal layer renaming, so we apply deltas directly to weights.
            # W_new = W_base + (lora_B @ lora_A) * scaling
            # ----------------------------------------------------------------
            import json
            from safetensors.torch import load_file

            logger.info("8-bit path: Adapter config found — loading fine-tuned model...")
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path")
            lora_r     = adapter_cfg["r"]
            lora_alpha = adapter_cfg["lora_alpha"]
            scaling    = lora_alpha / lora_r
            logger.info(f"Base model: {base_model_name} | LoRA r={lora_r}, alpha={lora_alpha}, scaling={scaling}")

            # Step 1: Load base model in 8-bit
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name    = base_model_name,
                max_seq_length= args.max_seq_length,
                load_in_4bit  = False,
                load_in_8bit  = True,
                dtype         = None,
                device_map    = device_map,
            )

            # Step 2: Parse adapter weights, strip Unsloth's prefix
            # Key format: 'base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight'
            # After stripping 'base_model.model.' → 'model.layers.X.self_attn.q_proj'
            logger.info(f"8-bit path: Loading and applying LoRA adapters from {model_path}...")
            adapter_weights = load_file(
                os.path.join(model_path, "adapter_model.safetensors"), device="cpu"
            )

            PREFIX = "base_model.model."
            lora_map = {}
            for key, tensor in adapter_weights.items():
                if not key.startswith(PREFIX):
                    continue
                stripped = key[len(PREFIX):]
                if ".lora_A.weight" in stripped:
                    module_path = stripped.replace(".lora_A.weight", "")
                    lora_map.setdefault(module_path, {})["lora_A"] = tensor
                elif ".lora_B.weight" in stripped:
                    module_path = stripped.replace(".lora_B.weight", "")
                    lora_map.setdefault(module_path, {})["lora_B"] = tensor

            # Step 3: Apply deltas directly to base model weights
            model_state = dict(model.named_parameters())
            applied, skipped = 0, 0
            with torch.no_grad():
                for module_path, lora_tensors in lora_map.items():
                    weight_key = f"{module_path}.weight"
                    if weight_key not in model_state:
                        logger.warning(f"Skipping — not found in model: {weight_key}")
                        skipped += 1
                        continue
                    lora_A = lora_tensors["lora_A"].to(torch.float32)
                    lora_B = lora_tensors["lora_B"].to(torch.float32)
                    delta  = (lora_B @ lora_A) * scaling
                    param  = model_state[weight_key]
                    param.data += delta.to(param.dtype).to(param.device)
                    applied += 1

            logger.info(f"LoRA deltas applied — Applied: {applied}, Skipped: {skipped}")

        else:
            # ----------------------------------------------------------------
            # 8-bit Base model path: No adapters — load directly via Unsloth
            # Used by eval_base.py where model_path is a HuggingFace model ID
            # ----------------------------------------------------------------
            logger.info("8-bit path: No adapter config found — loading as plain base model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name    = model_path,
                max_seq_length= args.max_seq_length,
                load_in_4bit  = False,
                load_in_8bit  = True,
                dtype         = None,
                device_map    = device_map,
            )

    elif getattr(args, "full_precision", False):
        # ----------------------------------------------------------------
        # bf16 full precision path — no quantization
        # ----------------------------------------------------------------
        PHI4_REASONING_VOCAB_SIZE = 200064
        use_phi4 = getattr(args, 'train_phi4', False)

        if use_phi4 and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Phi-4 reasoning special path:
            # resize_token_embeddings corrupts CUDA state on multi-GPU models.
            # Fix: load on single GPU → resize → apply deltas → dispatch to multi-GPU.
            import json
            from safetensors.torch import load_file

            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path")
            lora_r     = adapter_cfg["r"]
            lora_alpha = adapter_cfg["lora_alpha"]
            scaling    = lora_alpha / lora_r
            logger.info(f"Full precision Phi-4 path: base={base_model_name} | "
                        f"LoRA r={lora_r}, alpha={lora_alpha}, scaling={scaling}")

            # Step 1: Load base model on a SINGLE GPU to avoid multi-GPU resize issues
            logger.info("Loading base model on single GPU for safe resize...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name    = base_model_name,
                max_seq_length= args.max_seq_length,
                load_in_4bit  = False,
                load_in_8bit  = False,
                dtype         = torch.bfloat16,
                device_map    = {"": 0},
            )

            # Step 2: Resize embeddings on single GPU (safe)
            embed_size = model.get_input_embeddings().weight.shape[0]
            if embed_size < PHI4_REASONING_VOCAB_SIZE:
                logger.info(f"Resizing embeddings from {embed_size} to {PHI4_REASONING_VOCAB_SIZE}")
                model.resize_token_embeddings(PHI4_REASONING_VOCAB_SIZE)

            # Step 3: Load adapter weights and apply
            logger.info(f"Applying LoRA deltas from {model_path}")
            adapter_weights = load_file(
                os.path.join(model_path, "adapter_model.safetensors"), device="cpu"
            )

            PREFIX = "base_model.model."
            lora_map = {}
            full_weight_map = {}
            for key, tensor in adapter_weights.items():
                if not key.startswith(PREFIX):
                    continue
                stripped = key[len(PREFIX):]
                if ".lora_A.weight" in stripped:
                    module_path = stripped.replace(".lora_A.weight", "")
                    lora_map.setdefault(module_path, {})["lora_A"] = tensor
                elif ".lora_B.weight" in stripped:
                    module_path = stripped.replace(".lora_B.weight", "")
                    lora_map.setdefault(module_path, {})["lora_B"] = tensor
                else:
                    # Non-LoRA full weights (e.g. embed_tokens, lm_head after resize)
                    full_weight_map[stripped] = tensor

            model_state = dict(model.named_parameters())

            # Step 3a: Apply non-LoRA full weights (embed_tokens, lm_head)
            applied_full = 0
            with torch.no_grad():
                for weight_key, tensor in full_weight_map.items():
                    if weight_key in model_state:
                        param = model_state[weight_key]
                        param.data.copy_(tensor.to(param.dtype).to(param.device))
                        applied_full += 1
                        logger.info(f"  Loaded full weight: {weight_key} {list(tensor.shape)}")
                    else:
                        logger.warning(f"  Skipping full weight — not found: {weight_key}")

            # Step 3b: Apply LoRA deltas: W_new = W_base + (lora_B @ lora_A) * scaling
            applied_lora, skipped = 0, 0
            with torch.no_grad():
                for module_path, lora_tensors in lora_map.items():
                    weight_key = f"{module_path}.weight"
                    if weight_key not in model_state:
                        logger.warning(f"Skipping — not found in model: {weight_key}")
                        skipped += 1
                        continue
                    lora_A = lora_tensors["lora_A"].to(torch.float32)
                    lora_B = lora_tensors["lora_B"].to(torch.float32)
                    delta  = (lora_B @ lora_A) * scaling
                    param  = model_state[weight_key]
                    param.data += delta.to(param.dtype).to(param.device)
                    applied_lora += 1

            logger.info(f"Applied — Full weights: {applied_full}, "
                        f"LoRA deltas: {applied_lora}, Skipped: {skipped}")

            # Step 4: Dispatch to multi-GPU if requested
            if args.multi_gpu and torch.cuda.device_count() > 1:
                from accelerate import dispatch_model, infer_auto_device_map
                logger.info(f"Dispatching model across {torch.cuda.device_count()} GPUs...")
                new_device_map = infer_auto_device_map(model)
                model = dispatch_model(model, device_map=new_device_map)
        else:
            logger.info("Full precision path: Loading model in bf16, no quantization...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name    = model_path,
                max_seq_length= args.max_seq_length,
                load_in_4bit  = False,
                load_in_8bit  = False,
                dtype         = torch.float32,
                device_map    = device_map,
            )

    else:
        # ----------------------------------------------------------------
        # 4-bit path: Original working function — Unsloth handles everything
        # ----------------------------------------------------------------
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name    = model_path,
            max_seq_length= args.max_seq_length,
            load_in_4bit  = args.load_in_4bit,
            load_in_8bit  = False,
            dtype         = None,
            device_map    = device_map,
        )

    # Phi-4 reasoning uses an extended vocabulary (200064 vs base 100352).
    # Resize embeddings to match the fine-tuned checkpoint if needed.
    PHI4_REASONING_VOCAB_SIZE = 200064
    if getattr(args, 'train_phi4', False):
        embed_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"[Phi-4] Current embedding size: {embed_size}, required: {PHI4_REASONING_VOCAB_SIZE}")
        if embed_size < PHI4_REASONING_VOCAB_SIZE:
            logger.info(f"Resizing model embeddings from {embed_size} to {PHI4_REASONING_VOCAB_SIZE}")
            model.resize_token_embeddings(PHI4_REASONING_VOCAB_SIZE)
            logger.info(f"Embedding resize complete. New size: {model.get_input_embeddings().weight.shape[0]}")

    # When TORCHDYNAMO_DISABLE=1 is set (required for Phi-4 LongRoPE),
    # Unsloth's compiled cache can't manage dtype casting automatically.
    # Force bf16 only for full_precision path — do NOT cast for 4-bit/8-bit
    # as that would destroy quantized weights.
    # if getattr(args, "full_precision", False):
    #     model = model.to(torch.bfloat16)

    # Qwen3.5 returns a VLProcessor instead of a tokenizer.
    # Unwrap it so all downstream code (data_loader, trainer, eval)
    # gets a standard tokenizer that won't try to parse text as images.
    if hasattr(tokenizer, 'tokenizer'):
        logger.info(f"Unwrapping {type(tokenizer).__name__} → using inner tokenizer")
        tokenizer = tokenizer.tokenizer
    elif "Processor" in type(tokenizer).__name__:
        logger.warning(f"Tokenizer {type(tokenizer).__name__} looks like a processor but missing .tokenizer attribute.")

    return model, tokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import torch

# def _load_model(model_path, args, logger):
#     device_map = "auto" if args.multi_gpu else {"": 0}
#     load_in_8bit = getattr(args, "load_in_8bit", False)

#     # Read base model name from adapter config
#     import json
#     with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
#         adapter_cfg = json.load(f)
#     base_model_name = adapter_cfg.get("base_model_name_or_path")
#     logger.info(f"Base model from adapter_config: {base_model_name}")

#     # Step 1: Load base model
#     bnb_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

#     logger.info(f"Step 1: Loading base model [{base_model_name}] in {'8-bit' if load_in_8bit else 'bf16'}...")
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         quantization_config=bnb_config,
#         device_map=device_map,
#         torch_dtype=torch.bfloat16 if not load_in_8bit else None,
#         trust_remote_code=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#     # Step 2: Apply LoRA adapters
#     logger.info(f"Step 2: Applying LoRA adapters from [{model_path}]...")
#     model = PeftModel.from_pretrained(model, model_path, is_trainable=False)

#     model.eval()
#     logger.info("Model loaded successfully.")
#     return model, tokenizer



def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except (TypeError, OverflowError):
        return str(value)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _get_args():
    p = argparse.ArgumentParser(
        description="Evaluate the FINE-TUNED model on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_name", type=str, required=True,
                   help="Checkpoint name suffix.")
    p.add_argument("--model_name", type=str,
                   default="unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
                   help="Base model to load if checkpoint not found, or used to find checkpoint.")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--logging_dir", type=str, default="./logs")
    p.add_argument("--dataset_dir", type=str, default="./dataset")
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--heartbeat_tokens", type=int, default=50)
    p.add_argument("--pilot_size", type=int, default=100)
    p.add_argument("--pilot_seed", type=int, default=42)
    p.add_argument("--pilot_only", action="store_true")
    p.add_argument("--skip_pilot", action="store_true")
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
        args.load_in_4bit = True
    return args

# ---------------------------------------------------------------------------
# Streaming JSON writer
# ---------------------------------------------------------------------------

class _StreamingWriter:
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

def _fmt(seconds: float) -> str:
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
                out_path, label, ft_model_path, n_full=None):
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
    n = len(rows)
    w = len(str(n))
    sample_printed = False

    # -- Pass 1: Loss --
    logger.info(f"[{label}] Pass 1/2 — loss pass ({n} rows) …")
    losses = []
    t0 = time.time()
    for i, row in enumerate(rows):
        batch = _tokenize_for_loss(
            build_prompt(row), build_target(row),
            tokenizer, args.max_seq_length, args.max_new_tokens,
        )
        losses.append(_compute_sample_loss(model, batch, device))
        if (i + 1) % 25 == 0 or (i + 1) == n:
            valid = [l for l in losses if not math.isnan(l)]
            avg = sum(valid) / len(valid) if valid else float("nan")
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - (i + 1)) / rate if rate > 0 else float("nan")
            logger.info(f"  [{label}] loss [{i+1:{w}}/{n}] avg: {avg:.4f} ETA: {_fmt(eta)}")

    agg_loss = sum([l for l in losses if not math.isnan(l)]) / len([l for l in losses if not math.isnan(l)])
    agg_ppl  = math.exp(agg_loss) if math.isfinite(agg_loss) else float("nan")

    # -- Pass 2: Generation --
    logger.info(f"[{label}] Pass 2/2 — generation …")
    prompts = [build_prompt(r) + gen_suffix for r in rows]
    writer = _StreamingWriter(out_path, header={
        "run_name": args.run_name,
        "model_path": ft_model_path,
        "label": label.lower(),
        "n_samples": n,
    })

    t0 = time.time()
    completed = 0
    for start in range(0, n, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n)
        outputs = _generate_batch(model, tokenizer, prompts[start:end], args, device, logger, start)

        if not sample_printed and len(outputs) > 0:
            show_prompt_output_sample(prompts[start:end], outputs[0], tokenizer)
            sample_printed = True

        batch_records = []
        for j, (row, text) in enumerate(zip(rows[start:end], outputs)):
            rec = {k: _json_safe(v) for k, v in row.items()}
            rec["finetuned_model_output"] = text
            rec["per_sample_loss"] = losses[start + j]
            batch_records.append(rec)

        writer.write_batch(batch_records)
        completed += len(batch_records)
        elapsed = time.time() - t0
        rate = completed / elapsed
        eta_str = f" | ETA full: {_fmt(n_full / rate)}" if n_full else ""
        logger.info(f"  [{label}] batch [{completed}/{n}] {rate:.2f} rows/s{eta_str}")

    writer.close(footer={"aggregate_loss": agg_loss, "aggregate_ppl": agg_ppl})
    return agg_loss, agg_ppl, time.time() - t0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_base_eval(args, logger):
    # 1. Define the missing path
    ft_model_path = os.path.join(args.logging_dir, "checkpoints", args.run_name, "final_model")
    
    logger.info("=" * 60)
    logger.info(f"MODEL PATH: {ft_model_path}")
    
    # 2. Load model
    model, tokenizer = _load_model(ft_model_path, args, logger)
    device = next(model.parameters()).device
    model.eval()
    FastLanguageModel.for_inference(model)

    # 3. Load dataset
    all_rows = list(_load_raw_test(args.dataset_dir))
    n_total = len(all_rows)
    os.makedirs(args.logging_dir, exist_ok=True)

    pilot_path = os.path.join(args.logging_dir, f"{args.run_name}_finetuned_eval_pilot.json")
    full_path  = os.path.join(args.logging_dir, f"{args.run_name}_finetuned_eval.json")

    # Pilot Run
    if not args.skip_pilot:
        pilot_n = min(args.pilot_size, n_total)
        rng = random.Random(args.pilot_seed)
        pilot_rows = [all_rows[i] for i in sorted(rng.sample(range(n_total), pilot_n))]
        _eval_loop(model, tokenizer, pilot_rows, args, device, logger, pilot_path, "PILOT", ft_model_path, n_full=n_total)

    if args.pilot_only:
        return pilot_path, None

    # Full Run
    _eval_loop(model, tokenizer, all_rows, args, device, logger, full_path, "FULL", ft_model_path)
    
    del model
    torch.cuda.empty_cache()
    return pilot_path, full_path

if __name__ == "__main__":
    _args = _get_args()
    _logger = _setup_logger("eval_finetuned")
    run_base_eval(_args, _logger)