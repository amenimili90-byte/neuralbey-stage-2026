# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Phase 3 : entraînement du classifieur

Cahier des charges, section 4.2 : "Entraînement d'un modèle NLP de
classification de texte" + "Mise en place de seuils de confiance".

Modèle : CamemBERT-base, fine-tuning complet (même stack que le
classifieur Doctobot et le NER, pour cohérence).

Deux corrections apportées pour gérer le déséquilibre des classes
(77% "À modérer", seulement 6-10% "À bloquer") :
  1. Plafonnement de la classe majoritaire dans le train set (réduit aussi
     le temps d'entraînement, sinon ~30-40h sur CPU avec 23 370 lignes)
  2. Pondération de la loss (CrossEntropyLoss avec poids par classe) pour
     que le modèle ne soit pas juste "paresseux" et prédise toujours la
     classe majoritaire

Priorité donnée au RAPPEL sur "À bloquer" plutôt qu'à la précision : dans
un système de modération, laisser passer un contenu haineux (faux négatif)
est pire que bloquer par erreur un message inoffensif (faux positif), qui
peut être révisé par un modérateur humain.

Entrée  : processed/moderation/{train,val,test}.jsonl
Sortie  : models/moderation_classifier/final

Lancer avec :
    py train_moderation_classifier.py
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

DATA_DIR = Path("processed/moderation")
MODEL_NAME = "camembert-base"
OUTPUT_DIR = Path("models/moderation_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_PER_MAJORITY_CLASS = 4000  # plafond appliqué uniquement si la classe dépasse ce nombre

LABELS = ["Acceptable", "À modérer", "À bloquer"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

random.seed(42)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cap_majority_classes(records, max_per_class):
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    capped = []
    for label, items in by_label.items():
        if len(items) > max_per_class:
            random.shuffle(items)
            items = items[:max_per_class]
        capped.extend(items)
    random.shuffle(capped)
    return capped


def to_hf_dataset(records):
    return Dataset.from_list([
        {"text": r["text"], "label": LABEL2ID[r["label"]]} for r in records
    ])


class WeightedTrainer(Trainer):
    """Trainer avec CrossEntropyLoss pondérée par classe, pour compenser
    le déséquilibre restant après le plafonnement de la classe majoritaire."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    # Métriques spécifiques sur "À bloquer" (la classe critique)
    bloquer_id = LABEL2ID["À bloquer"]
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(
        labels, preds, labels=[bloquer_id], average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "recall_a_bloquer": r_b,
        "precision_a_bloquer": p_b,
    }


def main():
    print("Device détecté :", "cuda" if torch.cuda.is_available() else "cpu")

    print("\nChargement des données...")
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    val_records = load_jsonl(DATA_DIR / "val.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")

    print(f"  train (avant plafonnement) : {len(train_records)}")
    print("  ", dict(Counter(r["label"] for r in train_records)))

    train_records = cap_majority_classes(train_records, MAX_TRAIN_PER_MAJORITY_CLASS)
    print(f"  train (après plafonnement) : {len(train_records)}")
    print("  ", dict(Counter(r["label"] for r in train_records)))

    # Poids de classe = inversement proportionnels à la fréquence (après plafonnement)
    counts = Counter(r["label"] for r in train_records)
    total = sum(counts.values())
    weights = torch.tensor([
        total / (len(LABELS) * counts.get(label, 1)) for label in LABELS
    ], dtype=torch.float)
    print(f"\nPoids de classe appliqués à la loss : "
          f"{dict(zip(LABELS, weights.tolist()))}")

    train_ds = to_hf_dataset(train_records)
    val_ds = to_hf_dataset(val_records)
    test_ds = to_hf_dataset(test_records)

    print("\nChargement du tokenizer et du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="recall_a_bloquer",
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=weights,
    )

    print("\n=== Entraînement ===")
    trainer.train()

    print("\n=== Évaluation sur le test set ===")
    test_metrics = trainer.evaluate(test_ds)
    print(test_metrics)

    print("\nSauvegarde du modèle...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Modèle sauvegardé -> {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
