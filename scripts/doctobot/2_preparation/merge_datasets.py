"""
Doctobot - Phase 2 : Fusion des 3 sources en un dataset unifié
===================================================================

Usage :
    py merge_datasets.py
"""

import json
from pathlib import Path
from collections import Counter

PROCESSED_DIR = Path(__file__).parent / "processed"
OUT_PATH = PROCESSED_DIR / "doctobot_unified.jsonl"

SOURCE_FILES = [
    "animal_disease_prediction.jsonl",
    "pet_health_symptoms.jsonl",
    "peteval.jsonl",
]


def main():
    all_records = []
    for filename in SOURCE_FILES:
        path = PROCESSED_DIR / filename
        if not path.exists():
            print(f" Fichier manquant, ignoré : {path}")
            continue
        with open(path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f]
        all_records.extend(records)
        print(f"  {filename} : {len(records)} lignes")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n Dataset unifié : {len(all_records)} lignes -> {OUT_PATH}")

    # Statistiques globales
    source_counts = Counter(r["source"] for r in all_records)
    species_counts = Counter(r["species"] for r in all_records)
    n_with_symptoms = sum(1 for r in all_records if r["symptoms"])
    n_with_age = sum(1 for r in all_records if r["age"] is not None)

    print("\n--- Répartition par source ---")
    for src, count in source_counts.items():
        print(f"  {src} : {count}")

    print("\n--- Répartition par espèce (top 10) ---")
    for species, count in species_counts.most_common(10):
        print(f"  {species} : {count}")

    print(f"\nLignes avec au moins 1 symptôme : {n_with_symptoms} "
          f"({n_with_symptoms/len(all_records)*100:.1f}%)")
    print(f"Lignes avec âge connu           : {n_with_age} "
          f"({n_with_age/len(all_records)*100:.1f}%)")


if __name__ == "__main__":
    main()
