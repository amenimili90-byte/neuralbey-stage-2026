"""
Doctobot - Phase 2 : Transformation Animal Disease Prediction -> schéma commun
=================================================================================

Ce dataset est déjà structuré (Animal_Type, Age, Symptom_1-4, Disease_Prediction),
c'est donc la transformation la plus simple des 3 sources.

Point de vigilance traité ici : l'encodage redondant entre Symptom_1-4
(catégoriel) et les colonnes booléennes (Vomiting: Yes/No, etc.) — on ne
garde qu'UNE seule représentation : la liste dérivée de Symptom_1-4,
dédupliquée et normalisée.

Usage :
    py transform_animal_disease_prediction.py
"""

import json
import pandas as pd
from pathlib import Path

from normalization import normalize_species, normalize_symptom, derive_age_group

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

IN_PATH = RAW_DIR / "cleaned_animal_disease_prediction.csv"
OUT_PATH = PROCESSED_DIR / "animal_disease_prediction.jsonl"

SYMPTOM_COLUMNS = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]


def build_text(row) -> str:
    """Reconstruit une phrase en langage naturel à partir des colonnes
    structurées, pour permettre l'usage de cette ligne dans le RAG/NLU
    au même titre que les textes libres des 2 autres sources."""
    symptoms = [s for s in row["symptoms"] if s]
    symptoms_str = ", ".join(symptoms) if symptoms else "no specific symptoms reported"
    return (
        f"A {row['age']}-year-old {row['species']} presented with {symptoms_str}, "
        f"lasting {row['Duration']}."
    )


def transform_row(row, idx: int) -> dict:
    raw_symptoms = [row[col] for col in SYMPTOM_COLUMNS if pd.notna(row[col]) and row[col] != "No"]
    normalized_symptoms = sorted(set(normalize_symptom(s) for s in raw_symptoms))

    species = normalize_species(row["Animal_Type"])
    age = int(row["Age"]) if pd.notna(row["Age"]) else None

    record = {
        "id": f"ad_{idx:05d}",
        "source": "animal_disease_prediction",
        "species": species,
        "age": age,
        "age_group": derive_age_group(age, species),
        "symptoms": normalized_symptoms,
        "duration": row["Duration"] if pd.notna(row["Duration"]) else None,
        "disease_label": row["Disease_Prediction"] if pd.notna(row["Disease_Prediction"]) else None,
        "condition_category": None,
        "icd_category": [],
        "severity": "unknown",  # calculé dans une étape ultérieure (base de règles)
        "record_type": "structured_record",
    }
    record["text"] = build_text({**record, "Duration": record["duration"] or "an unspecified duration"})
    return record


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Lignes chargées : {len(df)}")

    records = [transform_row(row, idx) for idx, row in df.iterrows()]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" {len(records)} enregistrements transformés -> {OUT_PATH}")

    # Vérifications rapides
    n_unknown_species = sum(1 for r in records if r["species"] == "unknown")
    print(f"\nEspèces non reconnues : {n_unknown_species} ({n_unknown_species/len(records)*100:.1f}%)")
    print("\nExemple de résultat :")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
