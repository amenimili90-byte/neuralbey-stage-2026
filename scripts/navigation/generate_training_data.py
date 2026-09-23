# -*- coding: utf-8 -*-
"""
Module 3 - Chatbot de navigation - Génération des données d'entraînement

Lit intents_data.py (le fichier éditable) et produit les fichiers
train/val/test au format attendu par train_intent_classifier.py.

Relancez ce script à chaque fois que vous modifiez intents_data.py
(nouveaux intents, nouveaux exemples, vraies fonctionnalités Cheebo...).

Sortie : processed/navigation/{train,val,test}.jsonl
         processed/navigation/intents_meta.json (redirection + réponse par intent)

Lancer avec :
    py generate_training_data.py
"""

import json
import random
from pathlib import Path

from intents_data import INTENTS

OUT_DIR = Path("processed/navigation")
random.seed(42)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = [], [], []
    total_examples = 0

    print("Intents chargés :")
    for intent_id, data in INTENTS.items():
        examples = data["examples"]
        total_examples += len(examples)
        print(f"  {intent_id:22s}: {len(examples)} exemples")

        shuffled = examples[:]
        random.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        n_train = n - n_val - n_test

        for text in shuffled[:n_train]:
            train.append({"text": text, "label": intent_id})
        for text in shuffled[n_train:n_train + n_val]:
            val.append({"text": text, "label": intent_id})
        for text in shuffled[n_train + n_val:]:
            test.append({"text": text, "label": intent_id})

    print(f"\nTotal : {total_examples} exemples sur {len(INTENTS)} intents")
    print(f"  train: {len(train)} | val: {len(val)} | test: {len(test)}")

    if total_examples < 100:
        print("\n⚠️  Peu d'exemples au total. C'est attendu pour cette version "
              "bootstrap, mais le modèle sera plus fiable une fois "
              "intents_data.py enrichi avec de vraies questions Cheebo.")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        random.shuffle(split)
        out_path = OUT_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in split:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f" {name} -> {out_path}")

    # Métadonnées (redirection + message de guidage) pour l'inférence
    meta = {
        intent_id: {
            "redirection": data["redirection"],
            "reponse_type": data["reponse_type"],
        }
        for intent_id, data in INTENTS.items()
    }
    with open(OUT_DIR / "intents_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f" Métadonnées -> {OUT_DIR / 'intents_meta.json'}")


if __name__ == "__main__":
    main()
