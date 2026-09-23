# -*- coding: utf-8 -*-
"""
Module 3 - Chatbot de navigation - Entraînement du classifieur d'intents

Cahier des charges, section 4.3 : "Entraînement d'un modèle NLP orienté
assistance".

Même stack que les Modules 1 et 2 (CamemBERT, fine-tuning complet,
Trainer de Hugging Face) — cohérence technique sur tout le projet.

Dataset volontairement petit à ce stade (version bootstrap, voir
intents_data.py). Les métriques seront limitées jusqu'à l'enrichissement
avec de vraies questions Cheebo — c'est attendu, pas un échec.

Entrée  : processed/navigation/{train,val,test}.jsonl
Sortie  : models/navigation_classifier/final

Lancer avec :
    py train_intent_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from intents_data import INTENTS

DATA_DIR = Path("processed/navigation")
MODEL_NAME = "camembert-base"
OUTPUT_DIR = Path("models/navigation_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = list(INTENTS.keys())
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_hf_dataset(records):
    return Dataset.from_list([
        {"text": r["text"], "label": LABEL2ID[r["label"]]} for r in records
    ])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1_macro": f1, "precision_macro": precision, "recall_macro": recall}


def main():
    print("Device détecté :", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nIntents ({len(LABELS)}) : {LABELS}")

    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    val_records = load_jsonl(DATA_DIR / "val.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")
    print(f"\ntrain: {len(train_records)} | val: {len(val_records)} | test: {len(test_records)}")

    train_ds = to_hf_dataset(train_records)
    val_ds = to_hf_dataset(val_records)
    test_ds = to_hf_dataset(test_records)

    print("\nChargement du tokenizer et du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # --- Gel partiel du modèle ---
    # Avec un dataset aussi petit (quelques centaines de lignes), ré-entraîner
    # les 110M paramètres de CamemBERT en entier mène à un sous-apprentissage
    # sévère (constaté : loss bloquée au niveau du hasard lors du premier
    # essai). On gèle les embeddings + les 8 premières couches du Transformer
    # (sur 12), et on ne ré-entraîne que les 4 dernières couches + la tête de
    # classification — beaucoup moins de paramètres à apprendre, donc
    # beaucoup moins de données nécessaires pour converger correctement.
    N_LAYERS_TO_FREEZE = 8

    if not hasattr(model, "roberta"):
        raise AttributeError(
            f"Le modèle {MODEL_NAME} n'a pas l'attribut 'roberta' attendu. "
            f"Attributs disponibles : {list(model.__dict__.get('_modules', {}).keys())}. "
            f"Ajustez N_LAYERS_TO_FREEZE / le nom de l'attribut ci-dessous en conséquence."
        )

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    for layer in model.roberta.encoder.layer[:N_LAYERS_TO_FREEZE]:
        for param in layer.parameters():
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,  # gel partiel -> entraînement plus rapide, plus d'époques possibles
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n=== Entraînement ===")
    trainer.train()

    print("\n=== Évaluation sur le test set ===")
    test_metrics = trainer.evaluate(test_ds)
    print(test_metrics)

    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n Modèle sauvegardé -> {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
