import torch
import os
from unsloth import FastLanguageModel

def get_model_and_tokenizer(args, logger):
    """
    Loads the model with either 4-bit or 8-bit quantization.
    """
    logger.info(f"Loading Unsloth Model: {args.model_name}")
    
    # Safety Check: If user asks for 8-bit but points to a 4-bit pre-quantized model
    if args.load_in_8bit and "bnb-4bit" in args.model_name:
        logger.warning(f"⚠️ You requested 8-bit loading but the model name '{args.model_name}' implies 4-bit weights.")
        logger.warning("   Please use the base model name (e.g., 'unsloth/DeepSeek-R1-Distill-Qwen-14B') for 8-bit quantization.")
        # We don't crash, but we warn the user.

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 2. Force PyTorch to only see/use the GPU assigned to this process
    torch.cuda.set_device(local_rank)

    # FastLanguageModel handles both flags directly
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        device_map={"": local_rank},
        dtype = torch.bfloat16, # Auto-detect (Float16 or Bfloat16)
        load_in_4bit = args.load_in_4bit,
        load_in_8bit = args.load_in_8bit, # [NEW] Pass 8-bit flag
    )




    # Resize embeddings for Phi-4-reasoning models.
    # The Phi-4-reasoning tokenizer reports vocab_size=100352 via all standard APIs
    # (len(), .vocab_size, .get_vocab()), but its BPE merges produce token IDs up to ~200022.
    # We must hardcode the target size since no tokenizer API reflects the true max token ID.
    # Target: 200064 = next multiple of 64 after max_token_id(200022) + 1, for GPU alignment.
    PHI4_REASONING_VOCAB_SIZE = 200064
    if getattr(args, 'train_phi4', False):
        embed_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"[Phi-4] Current embedding size: {embed_size}, required: {PHI4_REASONING_VOCAB_SIZE}")
        if embed_size < PHI4_REASONING_VOCAB_SIZE:
            logger.info(f"Resizing model embeddings from {embed_size} to {PHI4_REASONING_VOCAB_SIZE}")
            model.resize_token_embeddings(PHI4_REASONING_VOCAB_SIZE)
            logger.info(f"Embedding resize complete. New size: {model.get_input_embeddings().weight.shape[0]}")





    quant_mode = "8-bit" if args.load_in_8bit else ("4-bit" if args.load_in_4bit else "Full Precision")
    logger.info(f"Model loaded successfully.")
    logger.info(f"Quantization Mode: {quant_mode}")
    logger.info(f"Max Sequence Length: {args.max_seq_length}")

    # Prepare for training
    FastLanguageModel.for_training(model)
    
    return model, tokenizer

def print_trainable_parameters(model, logger):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )