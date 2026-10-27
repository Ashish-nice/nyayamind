# import os
# import torch
# from datasets import load_dataset, load_from_disk
# from modules.data_formatter import format_deepseek_prompt, inspect_column_types

# def tokenize_function(example, tokenizer, max_len):
#     """
#     Tokenizes input and target, creating labels with masked inputs (-100).
#     """
#     # Tokenize Input (User Prompt)
#     input_enc = tokenizer(
#         example["input_text"],
#         add_special_tokens=False,
#         truncation=True,
#         max_length=max_len # Reserve space for output
#     )
    
#     remaining_len = max_len - len(input_enc["input_ids"])
    
#     # Tokenize Output (Assistant Response)
#     # target_enc = tokenizer(
#     #     example["target_text"],
#     #     add_special_tokens=False,
#     #     truncation=True,
#     #     max_length=max_len
#     # )
    
    
#     target_enc = tokenizer(
#         example["target_text"] + tokenizer.eos_token,
#         add_special_tokens=False,
#         truncation=True,
#         max_length=max(0, remaining_len)
#     )

#     # Combine
#     input_ids = input_enc["input_ids"] + target_enc["input_ids"]
#     attention_mask = [1] * len(input_ids)
    
#     # Mask input part for loss calculation (-100 is ignored by PyTorch CrossEntropy)
#     labels = [-100] * len(input_enc["input_ids"]) + target_enc["input_ids"]

#     # Final Truncation safety net
#     if len(input_ids) > max_len:
#         input_ids = input_ids[:max_len]
#         attention_mask = attention_mask[:max_len]
#         labels = labels[:max_len]

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

# def sanity_check_dataset(dataset, tokenizer, logger, num_samples=5):
#     """
#     Ensures that tokens are valid within the model's vocabulary.
#     """
#     logger.info(f"--- Running Sanity Check on {num_samples} samples ---")
#     vocab_size = len(tokenizer)
#     sanity_passed = True
    
#     for i in range(min(num_samples, len(dataset))):
#         sample = dataset[i]
#         labels = sample["labels"]
#         input_ids = sample["input_ids"]

#         # Check Vocabulary Bounds
#         invalid_tokens = [t for t in input_ids if t < 0 or t >= vocab_size]
#         if invalid_tokens:
#             sanity_passed = False
#             raise ValueError(f"Found tokens outside vocabulary bounds: {invalid_tokens}")

#         # Check Label Masking
#         # Ensure that we have at least some trainable tokens (not all -100)
#         trainable_tokens = [l for l in labels if l != -100]
#         if not trainable_tokens:
#              logger.warning(f"Sample {i} has NO trainable tokens (all masked)!")
        
#         logger.info(f"Sample {i} passed. Total len: {len(input_ids)}, Trainable tokens: {len(trainable_tokens)}")

#     if sanity_passed:
#         logger.info("✅ Sanity Check Passed: All tokens are valid.")

# def load_and_process_datasets(args, tokenizer, logger):
#     """
#     Main entry point for data loading.
#     1. Checks for cached tokenized data.
#     2. If not found, loads CSVs -> Formats -> Tokenizes.
#     3. Saves cache.
#     4. Runs sanity checks.
#     """
#     cache_path = os.path.join(args.dataset_dir, f"tokenized")
    
#     # --- 1. Load from Cache if Exists ---
#     if os.path.exists(cache_path):
#         logger.info(f"Loading cached tokenized dataset from: {cache_path}")
#         tokenized_datasets = load_from_disk(cache_path)
#         sanity_check_dataset(tokenized_datasets["train"], tokenizer, logger)
#         return tokenized_datasets

#     # --- 2. Load Raw Data ---
#     logger.info("Cache not found. Loading raw CSV files...")
#     data_files = {
#         "train": os.path.join(args.dataset_dir, "train.csv"),
#         "validation": os.path.join(args.dataset_dir, "val.csv"),
#         "test": os.path.join(args.dataset_dir, "test.csv")
#     }
    
#     # Filter only existing files
#     data_files = {k: v for k, v in data_files.items() if os.path.exists(v)}
#     raw_datasets = load_dataset("csv", data_files=data_files)
    
#     # --- 3. Inspect Columns ---
#     inspect_column_types(raw_datasets, logger)

#     # --- 4. Format & Tokenize ---
#     logger.info("Formatting and Tokenizing dataset...")
    
#     # Apply Prompt Template
#     formatted_datasets = raw_datasets.map(format_deepseek_prompt, num_proc=1)
    
#     # Tokenize
#     tokenized_datasets = formatted_datasets.map(
#         lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
#         batched=False,
#         remove_columns=raw_datasets["train"].column_names, # Remove raw text columns
#         num_proc=os.cpu_count()
#     )

#     # Set Format for PyTorch
#     tokenized_datasets.set_format("torch")

#     # --- 5. Save to Disk ---
#     logger.info(f"Saving tokenized dataset to {cache_path}")
#     tokenized_datasets.save_to_disk(cache_path)

#     # --- 6. Final Sanity Check ---
#     if "train" in tokenized_datasets:
#         sanity_check_dataset(tokenized_datasets["train"], tokenizer, logger)

#     return tokenized_datasets


import os
import torch
from datasets import load_dataset, load_from_disk
# from modules.data_formatter import format_deepseek_prompt, inspect_column_types
from modules.data_formatter import format_deepseek_prompt, format_phi4_prompt, format_qwen3_prompt, inspect_column_types
import torch.distributed as dist

# def tokenize_function(example, tokenizer, max_len):
#     """
#     Tokenizes input and target, creating labels with masked inputs (-100).
#     """
#     # Tokenize Input (User Prompt)
#     input_enc = tokenizer(
#         example["input_text"],
#         add_special_tokens=False,
#         truncation=True,
#         max_length=max_len # Reserve space for output
#     )
    
#     remaining_len = max_len - len(input_enc["input_ids"])
    
#     # Tokenize Output (Assistant Response)
#     target_enc = tokenizer(
#         example["target_text"] + tokenizer.eos_token,
#         add_special_tokens=False,
#         truncation=True,
#         max_length=max(0, remaining_len)
#     )

#     # Combine
#     input_ids = input_enc["input_ids"] + target_enc["input_ids"]
#     attention_mask = [1] * len(input_ids)
    
#     # Mask input part for loss calculation (-100 is ignored by PyTorch CrossEntropy)
#     labels = [-100] * len(input_enc["input_ids"]) + target_enc["input_ids"]

#     # Final Truncation safety net
#     if len(input_ids) > max_len:
#         input_ids = input_ids[:max_len]
#         attention_mask = attention_mask[:max_len]
#         labels = labels[:max_len]

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

def tokenize_function(example, tokenizer, max_seq_len, max_new_tokens=4096):
    """
    Tokenizes input and target with "Target Priority".
    1. Encode Target -> Truncate to max_new_tokens.
    2. Encode Input -> Truncate to (max_seq_len - len(Target)).
    """
    
    # --- 1. Tokenize Target (Response) ---
    # Add EOS token so the model learns to stop
    target_enc = tokenizer(
        example["target_text"] + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True, 
        max_length=max_new_tokens  # Hard cap on answer length
    )
    
    target_ids = target_enc["input_ids"]
    len_target = len(target_ids)

    # --- 2. Tokenize Input (Prompt) ---
    # Give the input ALL the remaining space
    len_input_space = max_seq_len - len_target
    
    input_enc = tokenizer(
        example["input_text"],
        add_special_tokens=False,
        truncation=True,
        max_length=len_input_space 
    )
    
    input_ids = input_enc["input_ids"]

    # --- 3. Combine ---
    full_input_ids = input_ids + target_ids
    
    # Attention Mask (1 for real tokens)
    attention_mask = [1] * len(full_input_ids)
    
    # Labels (-100 for input, actual IDs for target)
    labels = [-100] * len(input_ids) + target_ids

    return {
        "input_ids": full_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def sanity_check_dataset(dataset, tokenizer, logger, args, num_samples=5):
    """
    Ensures that tokens are valid within the model's vocabulary.
    Logs decoded tokens for debugging prompt templates and dataset issues.
    """
    logger.info(f"--- Running Sanity Check on {num_samples} samples ---")
    
    # Determine if this model has an extended vocabulary
    is_extended_vocab_model = getattr(args, 'model_short_name', '') in ['Phi-4-reasoning']
    
    # Use the maximum possible vocab size to account for added/special tokens
    vocab_size = max(len(tokenizer), getattr(tokenizer, 'vocab_size', 0))
    if hasattr(tokenizer, 'added_tokens_encoder') and tokenizer.added_tokens_encoder:
        vocab_size = max(vocab_size, max(tokenizer.added_tokens_encoder.values()) + 1)
    logger.info(f"Effective vocabulary size for sanity check: {vocab_size}")
    if is_extended_vocab_model:
        logger.info(f"Model '{args.model_short_name}' has extended vocabulary - will warn instead of crash for out-of-range tokens.")



    sanity_passed = True
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        labels = sample["labels"]
        input_ids = sample["input_ids"]

        # # Check Vocabulary Bounds
        # invalid_tokens = [t for t in input_ids if t < 0 or t >= vocab_size]
        # if invalid_tokens:
        #     sanity_passed = False
        #     raise ValueError(f"Found tokens outside vocabulary bounds: {invalid_tokens}")


        # Check for negative token IDs (always invalid)
        invalid_tokens = [t for t in input_ids if t < 0]
        if invalid_tokens:
            sanity_passed = False
            raise ValueError(f"Found negative token IDs: {invalid_tokens}")

        
        # Check for tokens exceeding reported vocab size
        out_of_range = [t for t in input_ids if t >= vocab_size]
        if out_of_range:
            # Log the decoded tokens so we can inspect what they represent
            unique_oor = list(set(int(t) for t in out_of_range))[:20]  # Show up to 20 unique
            decoded_samples = []
            for tid in unique_oor:
                try:
                    decoded = tokenizer.decode([tid])
                    decoded_samples.append(f"  ID {tid} -> '{decoded}'")
                except Exception:
                    decoded_samples.append(f"  ID {tid} -> [decode failed]")
            decoded_log = "\n".join(decoded_samples)
            
            if is_extended_vocab_model:
                logger.warning(
                    f"Sample {i}: {len(out_of_range)} tokens exceed reported vocab_size ({vocab_size}). "
                    f"Max token ID: {max(out_of_range)}. This is expected for {args.model_short_name}.\n"
                    f"Decoded out-of-range tokens (up to 20 unique):\n{decoded_log}"
                )
            else:
                logger.error(
                    f"Sample {i}: {len(out_of_range)} tokens exceed vocab_size ({vocab_size}).\n"
                    f"Decoded out-of-range tokens (up to 20 unique):\n{decoded_log}"
                )
                sanity_passed = False
                raise ValueError(
                    f"Found {len(out_of_range)} tokens outside vocabulary bounds. "
                    f"Max token ID: {max(out_of_range)}, vocab_size: {vocab_size}. "
                    f"Check your prompt template and dataset."
                )


        # Check Label Masking
        # Ensure that we have at least some trainable tokens (not all -100)
        trainable_tokens = [l for l in labels if l != -100]
        if not trainable_tokens:
             logger.warning(f"Sample {i} has NO trainable tokens (all masked)!")
        

        # Log decoded sample for debugging (first 200 tokens)
        if i == 0:
            try:
                decoded_input = tokenizer.decode(input_ids[:200], skip_special_tokens=False)
                logger.info(f"Sample {i} first 200 tokens decoded:\n{decoded_input}")
            except Exception as e:
                logger.warning(f"Could not decode sample {i}: {e}")
        

        logger.info(f"Sample {i} passed. Total len: {len(input_ids)}, Trainable tokens: {len(trainable_tokens)}")

    if sanity_passed:
        logger.info("✅ Sanity Check Passed: All tokens are valid.")


def load_and_process_datasets(args, tokenizer, logger):
    """
    Main entry point for data loading with Distributed Data Parallel (DDP) support.
    Ensures only Rank 0 processes raw data while others wait at a barrier.
    """
    # 1. Identify process rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Define a unique cache path for this specific run config to avoid truncation mismatches
    
    # cache_path = os.path.join(args.dataset_dir, f"tokenized_cache_{args.max_seq_length}")
    if getattr(args, 'train_phi4', False):
        model_tag = "phi4"
    elif getattr(args, 'train_qwen3', False):
        model_tag = "qwen3"
    else:
        model_tag = "deepseek"
    cache_path = os.path.join(args.dataset_dir, f"tokenized_cache_{model_tag}_{args.max_seq_length}")

    # --- 2. Rank 0: The "Worker" Rank ---
    if local_rank == 0:
        if os.path.exists(cache_path):
            logger.info(f"Rank 0: Found existing tokenized cache at {cache_path}. Skipping processing.")
        else:
            logger.info("Rank 0: Cache not found. Beginning raw JSON processing...")
            
            # Identify source files
            data_files = {
                "train": os.path.join(args.dataset_dir, "train_v2.json"),
                "validation": os.path.join(args.dataset_dir, "val_v2.json"),
                "test": os.path.join(args.dataset_dir, "test_v2.json")
            }
            data_files = {k: v for k, v in data_files.items() if os.path.exists(v)}
            
            if not data_files:
                raise FileNotFoundError(f"CRITICAL: No JSON files found in {args.dataset_dir}")

            # Load raw data
            raw_datasets = load_dataset("json", data_files=data_files)
            
            # Format prompts (num_proc=1 to avoid internal DDP multiprocess conflicts)
            # logger.info("Rank 0: Formatting prompts...")
            # Select formatter based on model type
            if getattr(args, 'train_phi4', False):
                formatter_fn = format_phi4_prompt
                fn_kwargs = {"logger": logger, "tokenizer": tokenizer}
            elif getattr(args, 'train_qwen3', False):
                formatter_fn = format_qwen3_prompt
                fn_kwargs = {"logger": logger}
            else:
                formatter_fn = format_deepseek_prompt
                fn_kwargs = {"logger": logger}

            logger.info(f"Rank 0: Formatting prompts with {formatter_fn.__name__}...")
            formatted_datasets = raw_datasets.map(
                # format_deepseek_prompt, 
                 formatter_fn, 
                num_proc=1, 
                fn_kwargs=fn_kwargs, 
                # fn_kwargs={"logger": logger}, 
                load_from_cache_file=False
            )
            
            # Tokenize and truncate
            logger.info(f"Rank 0: Tokenizing data (Max Length: {args.max_seq_length})...")
            cols_to_remove = formatted_datasets["train"].column_names
            tokenized_datasets = formatted_datasets.map(
                lambda x: tokenize_function(x, tokenizer, args.max_seq_length, args.max_new_tokens),
                batched=False,
                remove_columns=cols_to_remove,
                num_proc=os.cpu_count() # CPU-heavy task
            )

            # Save finished product to disk
            tokenized_datasets.save_to_disk(cache_path)
            logger.info(f"Rank 0: Data processing complete. Saved to {cache_path}")

    # --- 3. THE BARRIER: All other GPUs wait here ---
    # Rank 1, 2, etc. will pause here until Rank 0 hits this line
    if dist.is_initialized():
        logger.info(f"Rank {local_rank}: Synchronization point - waiting for Rank 0...")
        dist.barrier() 
        logger.info(f"Rank {local_rank}: Barrier released. All processes proceeding.")

    # --- 4. All Ranks: Load Finished Data ---
    # Now that the barrier is passed, we know for a fact the directory exists
    tokenized_datasets = load_from_disk(cache_path)
    tokenized_datasets.set_format("torch")

    # --- 5. Final Sanity Check (Rank 0 only) ---
    if local_rank == 0 and "train" in tokenized_datasets:
        sanity_check_dataset(tokenized_datasets["train"], tokenizer, logger, args)

    return tokenized_datasets