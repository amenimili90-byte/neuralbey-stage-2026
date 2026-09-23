"""
Doctobot - Phase 4 (NLP) : Préparation des données pour le classifieur CamemBERT
======================================================================================

Cahier des charges - section 4.1 (Doctobot), fonctionnalité visée :
  "Analyse des symptômes décrits par l'utilisateur" + "Interaction
  conversationnelle en langage naturel (texte)"

Choix : on entraîne uniquement sur owner_observation (pas clinical_notes),
car c'est le seul sous-ensemble qui reflète le langage naturel réellement
attendu d'un utilisateur de Doctobot (cf. cahier des charges ci-dessus).
Les clinical_notes sont un jargon procédural non représentatif de l'usage
réel de l'application.

Usage :
    py prepare_camembert_data.py
"""

import json
import csv
from pathlib import Path
from collections import Counter
import random

PROCESSED_DIR = Path(__file__).parent / "processed"
IN_PATH = PROCESSED_DIR / "doctobot_unified_with_severity.jsonl"

OUT_DIR = PROCESSED_DIR / "camembert"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)  # reproductibilité (exigence de traçabilité du rapport de stage)

TRAIN_RATIO, VAL_RATIO = 0.8, 0.1  # le reste (0.1) va au test


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    # Filtre : uniquement pet_health_symptoms / owner_observation
    filtered = [
        r for r in records
        if r["source"] == "pet_health_symptoms" and r["record_type"] == "owner_observation"
    ]
    print(f"Lignes retenues (owner_observation) : {len(filtered)}")

    label_counts = Counter(r["condition_category"] for r in filtered)
    print("Répartition des labels :")
    for label, count in label_counts.most_common():
        print(f"  {label} : {count}")

    # Split stratifié simple : on mélange par label puis on découpe
    by_label = {}
    for r in filtered:
        by_label.setdefault(r["condition_category"], []).append(r)

    train, val, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        out_path = OUT_DIR / f"{name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for r in subset:
                writer.writerow([r["text"], r["condition_category"]])
        print(f" {name} : {len(subset)} lignes -> {out_path}")

    # Sauvegarde de la correspondance label <-> id (nécessaire pour l'entraînement)
    labels_sorted = sorted(label_counts.keys())
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    with open(OUT_DIR / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print(f"\n Correspondance labels -> {OUT_DIR / 'label2id.json'}")
    print(label2id)


if __name__ == "__main__":
    main()
