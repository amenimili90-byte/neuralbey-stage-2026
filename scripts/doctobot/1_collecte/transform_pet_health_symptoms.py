"""
Doctobot - Phase 2 : Transformation Pet Health Symptoms -> schéma commun
============================================================================

Contrairement à Animal Disease Prediction, ce dataset n'a que du texte
libre. L'espèce et les symptômes sont donc EXTRAITS du texte via des
règles/mots-clés (voir normalization.py) — c'est une solution heuristique
temporaire, pas un vrai NER entraîné. À remplacer/améliorer une fois un
modèle NLP entraîné (conformément au cahier des charges).

Usage :
    py transform_pet_health_symptoms.py
"""

import json
import pandas as pd
from pathlib import Path

from normalization import extract_species_from_text, extract_symptoms_from_text

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

IN_PATH = RAW_DIR / "pet_health_symptoms.csv"
OUT_PATH = PROCESSED_DIR / "pet_health_symptoms.jsonl"

RECORD_TYPE_MAP = {
    "Owner Observation": "owner_observation",
    "Clinical Notes": "clinical_notes",
}


def transform_row(row, idx: int) -> dict:
    text = row["text"]
    species = extract_species_from_text(text)
    symptoms = extract_symptoms_from_text(text)

    return {
        "id": f"ph_{idx:05d}",
        "source": "pet_health_symptoms",
        "text": text,
        "species": species,
        "age": None,
        "age_group": "unknown",
        "symptoms": symptoms,
        "duration": None,
        "disease_label": None,
        "condition_category": row["condition"],
        "icd_category": [],
        "severity": "unknown",
        "record_type": RECORD_TYPE_MAP.get(row["record_type"], "clinical_free_text"),
    }


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Lignes chargées : {len(df)}")

    records = [transform_row(row, idx) for idx, row in df.iterrows()]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" {len(records)} enregistrements transformés -> {OUT_PATH}")

    # Statistiques de qualité de l'extraction heuristique
    n_unknown_species = sum(1 for r in records if r["species"] == "unknown")
    n_no_symptoms = sum(1 for r in records if len(r["symptoms"]) == 0)
    avg_symptoms = sum(len(r["symptoms"]) for r in records) / len(records)

    print(f"\nEspèces non reconnues       : {n_unknown_species} "
          f"({n_unknown_species/len(records)*100:.1f}%)")
    print(f"Lignes sans symptôme détecté : {n_no_symptoms} "
          f"({n_no_symptoms/len(records)*100:.1f}%)")
    print(f"Nombre moyen de symptômes/ligne : {avg_symptoms:.2f}")

    print("\nExemple de résultat :")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))

    print("\n  Extraction heuristique par mots-clés — taux d'erreur attendu.")
    print("   À affiner avec un vrai modèle NER une fois entraîné.")


if __name__ == "__main__":
    main()
