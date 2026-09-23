# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Validation à grande échelle

Mesure l'effet réel du filet de correction (moderation_correction.py) sur
le vrai test set (2922 lignes, jamais utilisées pour concevoir les règles
de correction) — contrairement aux 15 phrases écrites à la main, qui
avaient été choisies pour cibler des problèmes déjà identifiés et ne
prouvent pas une amélioration générale.

Compare précision/rappel/F1 par classe, AVANT et APRÈS correction.

Lancer avec :
    py evaluate_correction_on_testset.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from moderation_correction import decide_label

MODEL_DIR = Path("models/moderation_classifier/final")
TEST_PATH = Path("processed/moderation/test.jsonl")
BATCH_SIZE = 32
LABELS = ["Acceptable", "À modérer", "À bloquer"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("Chargement du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    print(f"Chargement de {TEST_PATH}...")
    records = load_jsonl(TEST_PATH)
    print(f"  -> {len(records)} lignes")

    y_true = [r["label"] for r in records]
    y_pred_raw = []
    y_pred_corrected = []

    print("\nInférence sur le test set (par lots)...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                            max_length=128, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        for j, rec in enumerate(batch):
            pred_label = model.config.id2label[int(pred_ids[j])]
            probs_dict = {model.config.id2label[k]: probs[j][k].item() for k in range(probs.shape[1])}
            corrected_label, _ = decide_label(rec["text"], probs_dict)
            y_pred_raw.append(pred_label)
            y_pred_corrected.append(corrected_label)

        done = min(i + BATCH_SIZE, len(records))
        print(f"  {done}/{len(records)}", end="\r")

    print("\n\n" + "=" * 70)
    print("AVANT correction (sortie brute du modèle)")
    print("=" * 70)
    print(classification_report(y_true, y_pred_raw, labels=LABELS, zero_division=0))

    print("=" * 70)
    print("APRÈS correction (filet de sécurité appliqué)")
    print("=" * 70)
    print(classification_report(y_true, y_pred_corrected, labels=LABELS, zero_division=0))

    # Sauvegarde pour référence / documentation
    report_raw = classification_report(y_true, y_pred_raw, labels=LABELS,
                                        zero_division=0, output_dict=True)
    report_corrected = classification_report(y_true, y_pred_corrected, labels=LABELS,
                                               zero_division=0, output_dict=True)
    with open("models/moderation_classifier/correction_impact.json", "w", encoding="utf-8") as f:
        json.dump({"avant_correction": report_raw, "apres_correction": report_corrected},
                   f, ensure_ascii=False, indent=2)
    print(" Rapport détaillé sauvegardé -> models/moderation_classifier/correction_impact.json")


if __name__ == "__main__":
    main()
