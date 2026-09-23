# -*- coding: utf-8 -*-
"""
Phase 4 - NER (fin) - Entraînement du modèle d'extraction d'entités
(animal, symptôme, âge) - Cahier des charges section 4.1 :
"Création et entraînement d'un modèle NLP" + "Structuration d'un système
de questions/réponses"

Modèle : camembert-base + tête de token classification (BIO)
Même logique que train_camembert.py (le classifieur d'intention) :
même stack (transformers/torch), même dossier, faisable sur CPU.

Entrée  : processed/ner/{train,val,test}.jsonl (générés par prepare_ner_data.py)
Sortie  : models/doctobot_ner/ (modèle sauvegardé) + métriques

Lancer avec :
    py train_ner_camembert.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

DATA_DIR = Path("processed/ner")
MODEL_NAME = "camembert-base"
OUTPUT_DIR = Path("models/doctobot_ner")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_LIST = ["O", "B-SPECIES", "I-SPECIES", "B-SYMPTOM", "I-SYMPTOM", "B-AGE", "I-AGE"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_split(name):
    path = DATA_DIR / f"{name}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "tokens": rec["tokens"],
                "ner_tags": [LABEL2ID[t] for t in rec["ner_tags"]],
            })
    return Dataset.from_list(rows)


def align_labels_with_tokens(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # tokens spéciaux ([CLS], [SEP], padding)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            else:
                # sous-mot supplémentaire du même mot : on répète en I- si
                # c'était un B-, sinon on garde tel quel
                label = labels[word_id]
                label_str = ID2LABEL[label]
                if label_str.startswith("B-"):
                    label = LABEL2ID["I-" + label_str[2:]]
                label_ids.append(label)
            previous_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def main():
    print("Device détecté :", "cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("  Pas de GPU détecté, l'entraînement se fera sur CPU "
              "(comme pour le classifieur, ça reste faisable vu la taille du dataset).")

    print("Chargement des données...")
    raw_datasets = {
        "train": load_split("train"),
        "validation": load_split("val"),
        "test": load_split("test"),
    }
    for name, ds in raw_datasets.items():
        print(f"  {name}: {len(ds)} exemples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_datasets = {}
    for name, ds in raw_datasets.items():
        tokenized_datasets[name] = ds.map(
            lambda ex: align_labels_with_tokens(ex, tokenizer),
            batched=True,
            remove_columns=ds.column_names,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [ID2LABEL[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n=== Entraînement ===")
    trainer.train()

    print("\n=== Évaluation sur le test set ===")
    test_metrics = trainer.evaluate(tokenized_datasets["test"])
    print(test_metrics)

    print("\nSauvegarde du modèle...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f" Modèle NER sauvegardé -> {OUTPUT_DIR / 'final'}")
    print(f"Métriques -> {OUTPUT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
