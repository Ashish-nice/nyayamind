import os
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, EarlyStoppingCallback
from unsloth import FastLanguageModel
import json

def make_serializable(obj):
    """Recursively converts non-serializable parts of a config to strings."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif callable(obj) or hasattr(obj, '__call__'):
        return str(obj) # Convert functions/methods to their string representation
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

def get_data_collator(tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

def setup_model_for_training(model, args, logger):
    """
    Applies Unsloth's optimized LoRA adapters.
    Works for both 4-bit and 8-bit loaded models.
    """

    logger.info("Inspecting model.config before modifications...")
    logger.info({k: type(v) for k, v in model.config.to_dict().items()})


    if args.lora_ft:
        logger.info(f"Configuring LoRA adapters (Base Quantization: {'8-bit' if args.load_in_8bit else '4-bit'})...")
        
        # Qwen 3.5 adds out_proj as an additional LoRA target (per reference script)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        if getattr(args, 'train_qwen3', False):
            target_modules.append("out_proj")
            logger.info(f"[Qwen3] Added 'out_proj' to LoRA target modules")
        
        logger.info(f"LoRA target modules: {target_modules}")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, 
            target_modules = target_modules,
            lora_alpha = 16,
            lora_dropout = 0, 
            bias = "none",   
            use_gradient_checkpointing = "unsloth", 
            random_state = 3407,
            use_rslora = False,
            loftq_config = None, 
        )
        
        model.print_trainable_parameters()
    
    else:
        logger.info("LoRA flag is OFF. Proceeding with Full Fine-Tuning.")
    if args.load_in_4bit:
        model.neftune_noise_alpha = 5

    return model

def get_training_args(args):
    output_dir = os.path.join(args.logging_dir, "checkpoints", args.run_name)

    train_kwargs = dict(
        output_dir=output_dir,
        run_name=args.run_name,
        
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        
        bf16=args.bf16,
        fp16=not args.bf16,
        
        gradient_checkpointing=args.gradient_checkpointing, 

        report_to="tensorboard",
        logging_dir=os.path.join(args.logging_dir, "tb_logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,

        # validation loss
        eval_strategy="steps",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        eval_steps=args.logging_steps,
        greater_is_better=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # LR Scheduler
        lr_scheduler_type="cosine",

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_checkpoints,
        
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # adamw_8bit is compatible with 8-bit model training as well
        optim = "adamw_8bit", 
        ddp_find_unused_parameters=False,
    )

     # group_by_length was removed in transformers 5.x (Qwen3 env)
    if not getattr(args, 'train_qwen3', False):
        train_kwargs["group_by_length"] = True

    return TrainingArguments(**train_kwargs)

def run_training(args, model, tokenizer, dataset, logger):
    collator = get_data_collator(tokenizer)
    model = setup_model_for_training(model, args, logger)
    train_args = get_training_args(args)

    if args.load_in_8bit:
        # 1. Clean the main model config
        for k, v in list(model.config.to_dict().items()):
            try:
                json.dumps(v)
            except (TypeError, OverflowError):
                logger.warning(f"Cleaning model.config.{k}")
                setattr(model.config, k, make_serializable(v))
                
        # 2. Clean PEFT config if it exists
        if hasattr(model, "peft_config"):
            for adapter_name, p_config in model.peft_config.items():
                for k, v in list(p_config.__dict__.items()):
                    try:
                        json.dumps(v)
                    except (TypeError, OverflowError):
                        logger.warning(f"Cleaning peft_config.{k}")
                        setattr(p_config, k, make_serializable(v))

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # transformers 5.x renamed 'tokenizer' to 'processing_class'
    if getattr(args, 'train_qwen3', False):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    

    logger.info(f"Starting Training... (Quant: {'8-bit' if args.load_in_8bit else '4-bit'})")
    
    checkpoint = None
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint = args.resume_from_checkpoint
            logger.info(f"Resuming from explicit checkpoint: {checkpoint}")
        else:
            logger.error(f"Checkpoint path not found: {args.resume_from_checkpoint}")
            raise FileNotFoundError(f"CRITICAL: Checkpoint path not found: {args.resume_from_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    logger.info("Training complete. Saving model...")
    final_save_path = os.path.join(train_args.output_dir, "final_model")
    
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    return trainer