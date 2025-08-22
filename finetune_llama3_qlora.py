import os, time, gc, json, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import (
    LoraConfig, TaskType,
    get_peft_model, prepare_model_for_kbit_training
)
from datasets import load_dataset
from trl import SFTTrainer

# 1. Paths
ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA, "training_data_with_responses.jsonl")
OUTPUT_DIR = "llama3-8b-qlora-finetuned"

# 2. Hyperparams
MODEL_NAME     = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_R         = 16 
LORA_ALPHA     = LORA_R * 2 
LORA_DROPOUT   = 0.05
BATCH_SIZE     = 4
EPOCHS         = 3
LR             = 2e-4
WARMUP_RATIO   = 0.03
MAX_SEQ_LENGTH = 1024
EVAL_RATIO     = 0.1

def print_gpu_stats(stage=""):
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    print(f"[GPU {stage}] Allocated: {a:.2f} GB | Reserved: {r:.2f} GB")

def main():
    # GPU check
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required.")
    torch.cuda.empty_cache(); gc.collect()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print_gpu_stats("Before load")

    # Quant config
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("→ 4-bit quantization configured")

    # Tokenizer & model
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_auth_token=True
    )
    tok.pad_token   = tok.pad_token or tok.eos_token
    tok.padding_side= "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True,
        use_cache=False
    )
    print_gpu_stats("After load")

    # K-bit & checkpointing
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    print("→ Model ready for k-bit training")

    # LoRA
    peft_conf = LoraConfig(
        r=LORA_R,#16
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj"] 
    )
    model = get_peft_model(model, peft_conf)
    t, tot = model.get_nb_trainable_parameters()
    print(f"→ LoRA: {t:,}/{tot:,} trainable")

    # Dataset
    ds = load_dataset("json", data_files={"all":TRAIN_FILE})["all"]
    ds = ds.train_test_split(test_size=EVAL_RATIO, seed=42)

    def prep(ex):
        instr = ex["instruction"].strip()
        ctx   = json.dumps(ex["context"], ensure_ascii=False)
        rsp   = ex["response"].strip()
        txt = f"### Instruction:\n{instr}\n\n### Context:\n{ctx}\n\n### Response:\n{rsp}"
        return {"text": txt}

    train_ds = ds["train"].map(prep, remove_columns=ds["train"].column_names)
    eval_ds  = ds["test"].map(prep,  remove_columns=ds["test"].column_names)
    print(f"→ Train: {len(train_ds)} samples; Eval: {len(eval_ds)}")

    # TrainingArguments
    grad_accum = max(1, 16 // BATCH_SIZE) #4 steps
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        optim="paged_adamw_8bit", #adamw
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        max_grad_norm=0.3,
        lr_scheduler_type="constant",
        group_by_length=True,
        dataloader_pin_memory=False
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_conf,
        tokenizer=tok,
        args=args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False
    )

    # Train
    torch.cuda.empty_cache(); gc.collect()
    st = time.time()
    res= trainer.train()
    print(f"\nDone in {(time.time()-st)/60:.1f}min; loss={res.training_loss:.4f}")
    print_gpu_stats("After train")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

    # Sanity
    model.config.use_cache=True
    smp = "### Instruction:\nSummarize this blueprint in two sentences.\n\n### Response:\n"
    inp = tok(smp, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    response_tokens=out[0,inp["input_ids"].shape[-1]:]
    gen=tok.decode(response_tokens,skip_special_tokens=True).strip()
    print("\nGenerated:\n"+gen)

if __name__=="__main__":
    main()
