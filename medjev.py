"""MedJev: typed decisions from a LoRA-adapted Qwen3.5-4B.

A decision is a state, a question and a closed set of options. `decide` returns a probability for every option,
read from the model's next-token distribution (no sampling, no JSON parsing).
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER = Path(__file__).resolve().parent / "medjev-4b-lora"

SYSTEM = ("You are a decision model. You are given a state, a question and a list of options. "
          "Choose the single option that best answers the question and reply with that option id only.")


def load(device="cuda", adapter=ADAPTER, merge=True):
    """Load Qwen3.5-4B and apply the MedJev adapter. Pass adapter=None for the base model."""
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.bfloat16, device_map=device)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        if merge:
            model = model.merge_and_unload()
    return model.eval(), tok


def _messages(state, question, options):
    opts = [{"id": k, "description": v} for k, v in options.items()]
    user = json.dumps({"state": state, "question": question, "options": opts}, ensure_ascii=False)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def make_example(state, question, options, answer):
    """One fine-tuning example in the format the model was trained on. `options` maps option id -> description."""
    assert answer in options
    return {"messages": _messages(state, question, options) + [{"role": "assistant", "content": answer}]}


@torch.no_grad()
def decide(model, tok, state, question, options):
    """Return (best_option_id, {option_id: probability}). `state` is any JSON-serialisable evidence;
    `options` maps option id -> description. Option ids must start with different tokens."""
    first = {k: tok.encode(k, add_special_tokens=False)[0] for k in options}
    if len(set(first.values())) != len(first):
        raise ValueError("option ids must begin with distinct tokens (use e.g. A/B/C or clearly different words)")
    enc = tok.apply_chat_template(_messages(state, question, options), add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt", return_dict=True)
    logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits[0, -1].float()
    probs = torch.softmax(logits[list(first.values())], 0).tolist()
    out = dict(zip(first, probs))
    return max(out, key=out.get), out
