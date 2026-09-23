# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Calibration du seuil de décision

Cahier des charges, section 4.2 : "Mise en place de seuils de confiance".

Contrairement à la règle de mots-clés testée précédemment (qui a cassé le
rappel sur "À bloquer" en le faisant tomber à 0% sur le vrai test set),
cette approche est calibrée sur les VRAIES probabilités du modèle,
mesurées sur le validation set (jamais vu à l'entraînement).

Principe : au lieu de toujours prendre la classe la plus probable
(argmax), on abaisse le seuil de décision pour "À bloquer" — si le
modèle attribue au moins X% de probabilité à "À bloquer", on la retient,
même si une autre classe a un score brut légèrement supérieur. Ça permet
de choisir explicitement un compromis précision/rappel, au lieu de le
laisser au hasard de l'architecture du modèle.

Lancer avec :
    py calibrate_threshold.py
"""

import json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

MODEL_DIR = Path("models/moderation_classifier/final")
VAL_PATH = Path("processed/moderation/val.jsonl")
BATCH_SIZE = 32
TARGET_CLASS = "À bloquer"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("Chargement du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    target_id = {v: k for k, v in model.config.id2label.items()}[TARGET_CLASS]

    print(f"Chargement de {VAL_PATH}...")
    records = load_jsonl(VAL_PATH)
    print(f"  -> {len(records)} lignes")

    y_true_binary = np.array([1 if r["label"] == TARGET_CLASS else 0 for r in records])
    scores = np.zeros(len(records))

    print("\nInférence sur le validation set...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                            max_length=128, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        scores[i:i + len(batch)] = probs[:, target_id].numpy()
        done = min(i + BATCH_SIZE, len(records))
        print(f"  {done}/{len(records)}", end="\r")

    print("\n\nSeuil | Precision | Recall | F1")
    print("-" * 45)
    best_f1, best_threshold = 0, 0.5
    candidates = [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)]
    for threshold in candidates:
        y_pred_binary = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        marker = ""
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
        print(f"{threshold:.2f}  | {precision:9.1%} | {recall:6.1%} | {f1:.3f}")

    print(f"\n✅ Meilleur seuil (F1 maximal) : {best_threshold:.2f} (F1={best_f1:.3f})")
    print("Choisissez le seuil manuellement dans le tableau ci-dessus si vous "
          "préférez privilégier davantage le rappel (sécurité) ou la précision "
          "(moins de faux positifs) que ce que le F1 recommande par défaut.")

    with open("models/moderation_classifier/threshold_config.json", "w", encoding="utf-8") as f:
        json.dump({"target_class": TARGET_CLASS, "recommended_threshold": best_threshold,
                    "f1_at_threshold": best_f1}, f, indent=2)
    print(f"\n Sauvegardé -> models/moderation_classifier/threshold_config.json")


if __name__ == "__main__":
    main()
