"""Two typed decisions: a laboratory result and a financial headline.   python examples/quickstart.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medjev import decide, load

model, tok = load()

choice, probs = decide(
    model, tok,
    state={"test": "serum potassium", "value": "6.8 mmol/L", "reference_range": "3.5-5.0 mmol/L"},
    question="How should this laboratory result be categorized?",
    options={"normal": "Within the reference range.",
             "abnormal": "Outside the reference range but not immediately dangerous.",
             "critical": "Dangerously abnormal; needs immediate clinician notification."},
)
print(choice, {k: round(v, 3) for k, v in probs.items()})

choice, probs = decide(
    model, tok,
    state={"headline": "Acme Corp cuts full-year revenue guidance after weak Q3 demand"},
    question="What does this news imply for the outlook of the company it mentions?",
    options={"bullish": "Positive for the outlook.", "bearish": "Negative for the outlook.", "neutral": "No clear implication."},
)
print(choice, {k: round(v, 3) for k, v in probs.items()})
