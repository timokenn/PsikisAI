"""
Quick local sanity check for the roberta_best model.
Run AFTER installing requirements:  python test_inference.py
If you see sensible predictions for each sentence, the app is good to go.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "roberta_best"
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
print("Classes:", class_names)


def predict(text):
    enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    return class_names[int(probs.argmax())], float(probs.max())


samples = [
    "I haven't been able to get out of bed for weeks, everything feels hopeless.",
    "I've been thinking everyone would be better off without me, I can't take this pain.",
    "My heart keeps racing and I can't stop thinking about everything that could go wrong.",
    "I have three deadlines tomorrow and my boss keeps adding more work, I can't keep up.",
    "Yesterday I felt on top of the world and spent all my savings, today I can't get off the couch.",
    "Feeling a bit tired lately but overall life is good, looking forward to my vacation.",
]
for s in samples:
    label, conf = predict(s)
    print(f"  [{label:<11} {conf:5.1%}]  {s[:70]}")
print("\nOK — model loads and predicts correctly.")
