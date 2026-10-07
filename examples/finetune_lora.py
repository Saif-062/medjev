"""Fine-tune MedJev (or the base model) on your own typed decisions with LoRA.

Data: a JSONL file, one example per line, as produced by medjev.make_example:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "<option id>"}]}

    python examples/finetune_lora.py --data mydata.jsonl --out my-adapter                 # start from the base model
    python examples/finetune_lora.py --data mydata.jsonl --out my-adapter --init medjev-4b-lora   # continue from MedJev

Loss is computed on the answer tokens only. Defaults are deliberately gentle (one epoch, lr 5e-5).
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medjev import BASE, load

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--init", default=None, help="existing adapter directory to continue training")
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--accum", type=int, default=16); ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--rank", type=int, default=16)
a = ap.parse_args()
random.seed(0); torch.manual_seed(0)

model, tok = load(adapter=None)
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
if a.init:
    model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

data = []
for line in open(a.data):
    m = json.loads(line)["messages"]
    p = tok.apply_chat_template(m[:2], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    a_ids = tok(m[2]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) <= a.max_len: data.append((p_ids, a_ids))
print(f"{len(data)} examples", flush=True)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
total = max(1, a.epochs * len(data) // a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(s, total) / total))))
model.train(); step, run, t0 = 0, [], time.time()
for _ in range(a.epochs):
    random.shuffle(data)
    for i, (p, ans) in enumerate(data):
        ids = torch.tensor([p + ans], device=model.device); lab = torch.tensor([[-100] * len(p) + ans], device=model.device)
        loss = model(input_ids=ids, labels=lab).loss
        (loss / a.accum).backward(); run.append(loss.item())
        if (i + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} loss {sum(run)/len(run):.4f} {time.time()-t0:.0f}s", flush=True); run = []
model.save_pretrained(a.out); print("saved", a.out)
