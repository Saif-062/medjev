# MedJev

Typed-decision adapter for Qwen3.5-4B, tuned on clinical question answering, financial news sentiment and
structured record-level workflow decisions.

Given evidence (`state`), a `question` and a closed set of `options`, MedJev returns a probability for every option,
read from the next-token distribution in a single forward pass. There is no generation loop and no output parsing,
so a decision can be used directly as a branch condition. The adapter is a rank-16 LoRA (27.9M trainable parameters,
56 MB). Base weights are fetched from Hugging Face at load time.

## Results

Accuracy under the same typed-decision prompt for the base model and for MedJev. The difference is a paired
estimate over identical items, with its 95% confidence interval.

| Task | Evaluation set | n | Qwen3.5-4B | MedJev | Difference |
|---|---|---:|---:|---:|---:|
| Clinical QA: MedQA (USMLE, 4 options) | official test, random subset | 300 | 71.7 | 73.3 | +1.7 ± 4.1 |
| Clinical QA: MedMCQA | official validation, held-out subset | 300 | 67.3 | 65.7 | −1.7 ± 4.3 |
| Biomedical literature QA: PubMedQA (yes / no / maybe) | labeled set, held-out | 400 | 79.5 | 80.0 | +0.5 ± 2.3 |
| Financial news sentiment (bearish / bullish / neutral) | twitter-financial-news-sentiment, validation, held-out | 300 | 66.3 | 85.3 | +19.0 ± 5.6 |
| Record-level workflow status (4-way) | synthetic practice charts, held-out patients | 292 | 72.6 | 99.3 | +26.7 ± 5.2 |

How to read these numbers:

- **Clinical QA is unchanged within noise.** The adaptation preserves the base model's medical knowledge; it does
  not add to it. All three clinical differences include zero.
- **Gains are in tasks whose labelling conventions the adapter was trained on**: financial sentiment and structured
  workflow decisions.
- **The workflow row is synthetic and in-distribution.** Labels follow a rule-defined scheme over generated
  records, and that data is not released. The score shows the model reproduces the scheme, not that it performs
  at that level on real records.
- **Scoring is not leaderboard-comparable.** Both columns use a zero-shot typed-decision prompt with thinking
  disabled, and the prediction is the option whose first token has the highest probability. There is no
  chain-of-thought and no few-shot context, so absolute numbers differ from published figures for the same benchmarks.
- No held-out item was used in training. MedQA and MedMCQA answer keys are known to contain errors, which bounds
  attainable accuracy; evaluation items were not re-verified.

## Quick start

```bash
git clone https://github.com/Saif-062/medjev.git && cd medjev
pip install -r requirements.txt
python examples/quickstart.py
```

The first run downloads the base model (about 9 GB). bf16 inference needs roughly 10 GB of GPU memory.

```python
from medjev import load, decide

model, tok = load()

choice, probs = decide(
    model, tok,
    state={"test": "serum potassium", "value": "6.8 mmol/L", "reference_range": "3.5-5.0 mmol/L"},
    question="How should this laboratory result be categorized?",
    options={
        "normal": "Within the reference range.",
        "abnormal": "Outside the reference range but not immediately dangerous.",
        "critical": "Dangerously abnormal; needs immediate clinician notification.",
    },
)
# critical {'normal': 0.0, 'abnormal': 0.009, 'critical': 0.991}
```

`state` is any JSON-serialisable evidence. `options` maps an option id to a plain-language description; the
descriptions are part of the prompt, so write them as criteria. Option ids must begin with different tokens
(`A/B/C`, or clearly different words). `load(adapter=None)` returns the unmodified base model for comparison.

## Fine-tune on your own decisions

Build examples with `make_example`, then train a LoRA with the included script.

```python
import json
from medjev import make_example

rows = [make_example(state, question, options, answer) for ... in your_data]
open("mydata.jsonl", "w").write("\n".join(json.dumps(r) for r in rows))
```

```bash
python examples/finetune_lora.py --data mydata.jsonl --out my-adapter                       # from the base model
python examples/finetune_lora.py --data mydata.jsonl --out my-adapter --init medjev-4b-lora # continue from MedJev
```

```python
model, tok = load(adapter="my-adapter")
```

Loss is computed on the answer tokens only. Defaults are conservative (one epoch, learning rate 5e-5, rank 16). Our
run of 3,444 examples took about 93 minutes on a single NVIDIA GB10. Hold out evaluation items before training,
and compare against the base model on your own data, since the gains above were task-specific.

## Training details

| | |
|---|---|
| Base model | Qwen3.5-4B (pinned revision `851bf6e`), bf16 |
| Method | LoRA, r = 16, α = 32, dropout 0.05, on attention, linear-attention and MLP projections |
| Data | 3,444 examples: MedQA 567, MedMCQA 702, PubMedQA 407, financial news sentiment 658, synthetic structured-record decisions 1,110 |
| Schedule | 1 epoch, 215 optimizer steps, effective batch 16, learning rate 5e-5 with cosine decay, max length 3,072 |
| Loss | answer tokens only |
| Key screening | Training answer keys were independently re-answered without seeing the key; disagreements were adjudicated blind. About 12% of reviewed items were removed and 25 were corrected. Reviewers were language models, not clinicians. |

Training used the training splits of the public datasets. The training set is not distributed.

## Intended use and limitations

- Research and prototyping. MedJev is **not a medical device** and has not been validated for diagnosis, treatment,
  triage or any patient-care decision. Outputs must not drive care without clinician review and local validation.
  Nothing here is financial advice.
- It is a 4B-parameter model and will make errors. Option probabilities are not guaranteed to be calibrated; this
  has not been evaluated.
- Evaluated only on the tasks above, in English. Open-ended generation, safety behaviour and performance on other
  populations or document types were not measured.
- Workflow-decision training data is synthetic.
- If you fine-tune on patient data, protecting that data and meeting the rules that apply to it is your
  responsibility.

## License and attribution

Code and adapter weights: Apache License 2.0 (see `LICENSE`). Built on
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Alibaba Cloud, Apache-2.0); base weights are not redistributed.

Training drew on MedQA-USMLE (CC BY-SA 4.0), MedMCQA (Apache-2.0), PubMedQA (MIT) and
twitter-financial-news-sentiment (MIT). Whether training on share-alike data places conditions on the resulting
weights is not settled; review the dataset licenses for your use. See `NOTICE`.
