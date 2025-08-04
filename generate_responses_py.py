"""generate_responses.py: fills `response` via Llama-3 and writes data/training_data_with_responses.jsonl"""

import os, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Paths
ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "data")
INP  = os.path.join(DATA, "training_data.jsonl")
OUTP = os.path.join(DATA, "training_data_with_responses.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # 2. Tokenizer & model (requires HF_TOKEN or `huggingface-cli login`)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_lade=True, use_auth_token=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True,
        use_cache=True
    )
    model.eval()

    # 3. Load prompts
    prompts = []
    with open(INP, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    # 4. Generate and write
    with open(OUTP, "w", encoding="utf-8") as fout:
        for ex in tqdm(prompts, desc="Generating"):
            instr = ex["instruction"].strip()
            ctx   = json.dumps(ex["context"], ensure_ascii=False)
            prompt = (
                "### Instruction:\n" + instr +
                "\n\n### Context:\n" + ctx +
                "\n\n### Response:\n"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            full     = tokenizer.decode(out[0], skip_special_tokens=True)
            response = full[len(prompt):].strip()
            ex["response"] = response
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} examples with responses to {OUTP}")

if __name__ == "__main__":
    main()