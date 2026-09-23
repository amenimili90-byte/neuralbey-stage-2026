"""
Doctobot - Phase 4 (NLP) : Entraînement du classifieur CamemBERT
=====================================================================

Cahier des charges - section 4.1 (Doctobot) :
  "Analyse des symptômes décrits par l'utilisateur" +
  "Création et entraînement d'un modèle NLP" (Travail du stagiaire)
Cahier des charges - section 5 (Technologies) :
  NLP : Hugging Face | Modèles : Classification de texte

Ce modèle prédit condition_category (5 classes : Digestive Issues,
Ear Infections, Mobility Problems, Parasites, Skin Irritations) à partir
du texte libre décrit par l'utilisateur.

Prérequis :
    pip install transformers torch scikit-learn datasets accelerate

Usage :
    py train_camembert.py
"""

import json
import time
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)

CAMEMBERT_DIR = Path(__file__).parent / "processed" / "camembert"
MODEL_OUT_DIR = Path(__file__).parent / "models" / "camembert_condition_classifier"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "camembert-base"


def main():
    # --- Détection GPU/CPU ---------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device détecté : {device}")
    if device == "cpu":
        print("  Pas de GPU détecté. L'entraînement sur CPU sera plus lent")
        print("   (attendre plusieurs dizaines de minutes selon ta machine).")
        print("   Si c'est trop lent, on peut basculer sur Google Colab (GPU gratuit).")

    # --- Chargement des labels -------------------------------------------
    with open(CAMEMBERT_DIR / "label2id.json", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    print(f"Nombre de classes : {num_labels} -> {list(label2id.keys())}")

    # --- Chargement des données ------------------------------------------
    dataset = load_dataset("csv", data_files={
        "train": str(CAMEMBERT_DIR / "train.csv"),
        "validation": str(CAMEMBERT_DIR / "val.csv"),
        "test": str(CAMEMBERT_DIR / "test.csv"),
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_encode(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=128)
        encoded["labels"] = [label2id[label] for label in batch["label"]]
        return encoded

    dataset = dataset.map(tokenize_and_encode, batched=True,
                           remove_columns=["text", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Modèle -----------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )

    # --- Métriques ----------------------------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    # --- Configuration de l'entraînement -----------------------------------
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8 if device == "cpu" else 16,
        per_device_eval_batch_size=8 if device == "cpu" else 16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",  # pas de wandb/tensorboard, évite une dépendance en plus
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Entraînement ---------------------------------------------------------
    print("\n--- Début de l'entraînement ---")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\n✅ Entraînement terminé en {elapsed/60:.1f} minutes")

    # --- Évaluation finale sur le jeu de test (jamais vu pendant l'entraînement) ---
    print("\n--- Évaluation sur le jeu de test ---")
    test_predictions = trainer.predict(dataset["test"])
    preds = np.argmax(test_predictions.predictions, axis=-1)
    labels = test_predictions.label_ids

    print(classification_report(
        labels, preds, target_names=[id2label[i] for i in range(num_labels)]
    ))

    # --- Sauvegarde du modèle final (nécessaire pour les livrables du cahier
    # des charges - section 6 : "Modèles NLP entraînés") -----------------------
    trainer.save_model(str(MODEL_OUT_DIR / "final"))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR / "final"))
    print(f"\n Modèle final sauvegardé -> {MODEL_OUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
