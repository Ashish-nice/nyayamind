# =========================================
#          LOGIN TO HUGGINGFACE HUB
# =========================================

import os
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
# Force Unsloth to use a local temp folder instead of trying to sync across GPUs
os.environ["UNSLOTH_COMPILE_USE_TEMP"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# Disable the specific compiler logic that is causing the broadcast crash
os.environ["UNSLOTH_USE_COMPILED"] = "0"
import torch
import torch.distributed as dist
from datetime import timedelta
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Only login on the main process (Rank 0) to avoid race conditions
if os.environ.get("LOCAL_RANK", "0") == "0":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

# =========================================
#               MAIN SCRIPT
# =========================================

from modules.arguments import get_args
from modules.logger import setup_logger
from modules.model_loader import get_model_and_tokenizer
from modules.data_loader import load_and_process_datasets
from modules.trainer import run_training

def main():
    # 0. Initialize Distributed Environment EARLY
    # This is critical so that dist.barrier() in data_loader works!
    if "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200)  # 2hr timeout for data processing barrier
            )
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)

    # 1. Collect Arguments
    args = get_args()

    # 2. Setup Logger
    logger = setup_logger(args)

    # 3. Test the Logger
    logger.info("Main execution started.")
    logger.info(f"Selected Model: {args.model_name}")

    # 4. Load Model & Tokenizer
    model, tokenizer = get_model_and_tokenizer(args, logger)

    # Qwen3.5 returns a VLProcessor instead of a tokenizer.
    # Unwrap it so all downstream code (data_loader, trainer, eval)
    # gets a standard tokenizer that won't try to parse text as images.
    if hasattr(tokenizer, 'tokenizer'):
        logger.info(f"Unwrapping {type(tokenizer).__name__} → using inner tokenizer")
        tokenizer = tokenizer.tokenizer
    
    # 5. [CRITICAL] Resize Embeddings Check
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resize needed: Tokenizer len ({len(tokenizer)}) > Model Embedding ({embedding_size})")
        model.resize_token_embeddings(len(tokenizer))


    # 6. Load & Process Data
    # Now that DDP is initialized, the barrier inside here will sync all GPUs
    dataset = load_and_process_datasets(args, tokenizer, logger)
    
    # Debug info only on Rank 0 to keep logs clean
    if os.environ.get("LOCAL_RANK", "0") == "0":
        logger.info(f"Training set size: {len(dataset['train'])}")
        sample = dataset["train"][5]
        logger.info("===== SAMPLE DEBUG =====")
        logger.info(f"Input text decoded: {tokenizer.decode([i for i in sample['input_ids'] if i != tokenizer.pad_token_id])}...")
        logger.info("========================")

    # 7. Start Training    
    if not args.inference:
        logger.info("Starting training phase...")
        run_training(args, model, tokenizer, dataset, logger)
    else:
        logger.info("--- Inference Mode Enabled: Skipping Training Phase ---")

    # 8. Evaluation Audit (Rank 0 only logic inside modules.eval)
    # if "test" in dataset:
    #     logger.info("Starting post-training evaluation on test set...")
    #     run_comparison_audit(
    #         args=args,
    #         tokenizer=tokenizer,
    #         logger=logger,
    #         test_dataset=dataset["test"],
    #         num_samples=len(dataset["test"])
    #     )
    # else:
    #     logger.warning("No test split found for evaluation.")

    # 9. Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()