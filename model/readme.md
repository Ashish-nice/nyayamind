## NyayaMind - Model 

The repository holds the model developed as part of the NyayaMind project, a framework for transparent legal reasoning and judgment prediction in the Indian legal system.

As part of the project we finetune various reasoning-oriented large language models to leverage the reasoning capabilities of these models for the legal judgment prediction task. The models are finetuned to follow a structured legal reasoning process which is grounded in legal issue, arguements of petitioner and arguments of respondents. 

We trained the below models using Supervised fine-tuning (SFT) adopting a parameter-efficient fine-tuning (PEFT) strategy based on LoRA and QLoRA. Unsloth framework was used for all the training pipelines.  

- DeepSeek-R1-Distill-Qwen-14B (4,8,16,32)-bit models
- Phi-4-mini-reasoning (16-bit)
- Phi-4-reasoning (16-bit)
- Qwen3.5-27B (16-bit)

The best performing model Qwen3.5-27B model along with the LoRA Adaptors are available at the [drive](https://drive.google.com/drive/folders/154VMFASsRmLDJ3zeLezUYVea2KPDU9tt?usp=sharing)

To run inference on a list of samples, with the state-of-the-art (SOTA) model run the following command: 

```bash
python inference.py --pilot_only  --run_name Qwen3_DDP_v14_final  --train_qwen3  --logging_dir ../logs  --dataset_dir ../dataset  --max_seq_length 32769   --max_new_tokens 8192  --eval_batch_size 2  --multi_gpu --full_precision
```


