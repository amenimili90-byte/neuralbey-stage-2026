# -*- coding: utf-8 -*-
"""
Module 3 - Chatbot de navigation - Calibration du seuil de confiance

Même logique que calibrate_threshold.py du Module 2 (Modération) : on
mesure, sur le validation set, la relation réelle entre confiance du
modèle et justesse de la prédiction — plutôt que de fixer un seuil au
hasard (0,35 s'est révélé mal calibré : le modèle a souvent raison avec
une confiance modeste, à cause du nombre de classes et de la petite
taille du dataset).

Pour chaque seuil testé, affiche :
  - Couverture : % des questions où le modèle est assez confiant pour répondre
  - Précision sur les réponses données : parmi les questions où il répond,
    quel pourcentage est correct

Lancer avec :
    py calibrate_navigation_threshold.py
"""

import json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("models/navigation_classifier/final")
VAL_PATH = Path("processed/navigation/val.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    records = load_jsonl(VAL_PATH)
    print(f"Validation set : {len(records)} exemples\n")

    confidences = []
    is_correct = []

    for rec in records:
        inputs = tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        pred_label = model.config.id2label[pred_id]
        confidence = probs[pred_id].item()

        confidences.append(confidence)
        is_correct.append(pred_label == rec["label"])

    confidences = np.array(confidences)
    is_correct = np.array(is_correct)

    print(f"{'Seuil':>6s} | {'Couverture':>10s} | {'Précision (accepté)':>20s}")
    print("-" * 45)
    for threshold in [round(x, 2) for x in np.arange(0.10, 0.55, 0.05)]:
        accepted = confidences >= threshold
        coverage = accepted.mean()
        if accepted.sum() > 0:
            precision = is_correct[accepted].mean()
        else:
            precision = float("nan")
        print(f"{threshold:>6.2f} | {coverage:>9.1%} | {precision:>19.1%}")

    print("\nChoisissez un seuil qui garde une bonne couverture (le chatbot "
          "répond souvent) sans trop sacrifier la précision (les réponses "
          "données sont fiables). Éditez ensuite CONFIDENCE_THRESHOLD dans "
          "navigation_chatbot.py avec la valeur retenue.")


if __name__ == "__main__":
    main()
