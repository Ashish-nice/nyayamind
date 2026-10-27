import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Unsloth Finetuning Script (4-bit / 8-bit support)")

    # =========================
    # 1. System & Hardware
    # =========================
    sys_group = parser.add_argument_group("System & Hardware")
    sys_group.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs to utilize")
    sys_group.add_argument("--gpu_index", type=str, default="0", help="Specific GPU index")
    sys_group.add_argument("--seed", type=int, default=42, help="Random seed")
    sys_group.add_argument("--inference", action="store_true", 
                           help="If set, skips training and runs only the evaluation/audit.")
    sys_group.add_argument("--train_phi4", action="store_true",
                           help="Train Phi-4 reasoning models instead of DeepSeek. "
                                "Use with --model_name pointing to an Unsloth Phi-4 repo.")
    sys_group.add_argument("--train_qwen3", action="store_true",
                           help="Train Qwen 3.5 reasoning models instead of DeepSeek. "
                                "Use with --model_name pointing to an Unsloth Qwen3.5 repo.")

    # Changed type=bool to action="store_true" to fix JSON serialization error
    sys_group.add_argument("--bf16", action="store_true", help="Use bf16 precision (Recommended for Ampere+)")
    
    # Quantization Flags
    sys_group.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    sys_group.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")

    # =========================
    # 2. Model Configuration
    # =========================
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_name", type=str, required=False, 
                            default="unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
                             help="Model name. NOTE: If using 8-bit, use the base version: 'unsloth/DeepSeek-R1-Distill-Qwen-14B'")
    model_group.add_argument("--model_short_name", type=str, required=True, 
                             choices=["MST_7B_IT", "DeepSeek-R1-Distill-Qwen-14B",
                                      "Phi-4-reasoning", "Phi-4-reasoning-plus",
                                      "Phi-4-mini-reasoning",
                                      "Qwen3.5-27B"], 
                             help="Short abbreviation for file naming")
    
    # Changed type=bool to action="store_true"
    model_group.add_argument("--lora_ft", action="store_true", help="Enable LoRA fine-tuning")

    # =========================
    # 3. Training Hyperparameters
    # =========================
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    train_group.add_argument("--epoch", type=int, default=1, help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # Changed type=bool to action="store_true"
    train_group.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    train_group.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of steps for warmup")
    train_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    train_group.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length")
    
    # =========================
    # 4. Logging & Saving
    # =========================
    log_group = parser.add_argument_group("Logging & Saving")
    log_group.add_argument("--run_name", type=str, required=True, help="Unique identifier for this run")
    log_group.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logs")
    log_group.add_argument("--logging_steps", type=int, default=50, help="Log every X steps")
    log_group.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X steps")
    log_group.add_argument("--save_total_checkpoints", type=int, default=3, help="Max checkpoints to keep")
    log_group.add_argument("--resume_from_checkpoint", type=str, default=None, 
                           help="Path to a specific checkpoint folder to resume from")
    # =========================
    # 5. Dataset parameters
    # =========================
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument("--dataset_dir", type=str, default="dataset/", help="Directory of the dataset")
    data_group.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens for target masking")

    args = parser.parse_args()
    
    # [Check] Enforce mutual exclusivity
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot set BOTH --load_in_4bit and --load_in_8bit. Choose one.")
    
    if args.train_phi4 and args.train_qwen3:
        raise ValueError("Cannot set BOTH --train_phi4 and --train_qwen3. Choose one.")
    
    # # Default to 4-bit if neither is specified (Safe default for Unsloth)
    # # Skip this default when using Phi-4 — let user choose explicitly
    # if not args.load_in_4bit and not args.load_in_8bit and not args.train_phi4:
    #     args.load_in_4bit = True

    return args