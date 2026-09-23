"""
Doctobot - Phase 2 : Transformation PetEVAL -> schéma commun
=================================================================

Utilise test_cleaned.jsonl (produit par parse_ner_column.py), qui contient
déjà les entités 'disease' et les catégories 'icd_label' correctement
parsées.

Choix de mapping :
  - Les entités 'disease' (NER) sont normalisées (abréviations dépliées,
    ex. 'oa' -> 'osteoarthritis') et placées dans le champ `symptoms`,
    car dans ce contexte de notes cliniques libres, ce sont les mentions
    de maladies/symptômes évoquées dans le texte, pas un diagnostic
    final unique -> pas de valeur unique pour `disease_label`.
  - `icd_category` reprend directement `icd_label` (déjà une taxonomie
    propre à PetEVAL, on ne la fusionne pas avec `condition_category`
    qui appartient à une autre taxonomie, Pet Health Symptoms).
  - `species` est extrait du texte libre (mêmes limites que pour
    Pet Health Symptoms : notes cliniques abrégées, taux d'échec attendu).

Usage :
    py transform_peteval.py
"""

import json
from pathlib import Path

from normalization import extract_species_from_text, normalize_disease_term

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

IN_PATH = RAW_DIR / "peteval_json" / "test_cleaned.jsonl"
OUT_PATH = PROCESSED_DIR / "peteval.jsonl"


def transform_row(row: dict, idx: int) -> dict:
    text = row["sentence"]
    species = extract_species_from_text(text)

    symptoms = sorted({
        normalize_disease_term(ent["entity"]) for ent in row.get("disease", [])
    })

    return {
        "id": f"pe_{idx:05d}",
        "source": "peteval",
        "text": text,
        "species": species,
        "age": None,
        "age_group": "unknown",
        "symptoms": symptoms,
        "duration": None,
        "disease_label": None,
        "condition_category": None,
        "icd_category": row.get("icd_label", []),
        "severity": "unknown",
        "record_type": "clinical_free_text",
    }


def main():
    records = []
    with open(IN_PATH, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            records.append(transform_row(row, idx))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" {len(records)} enregistrements transformés -> {OUT_PATH}")

    n_unknown_species = sum(1 for r in records if r["species"] == "unknown")
    n_no_symptoms = sum(1 for r in records if len(r["symptoms"]) == 0)
    n_no_icd = sum(1 for r in records if len(r["icd_category"]) == 0)

    print(f"\nEspèces non reconnues        : {n_unknown_species} "
          f"({n_unknown_species/len(records)*100:.1f}%)")
    print(f"Lignes sans symptôme (NER)   : {n_no_symptoms} "
          f"({n_no_symptoms/len(records)*100:.1f}%)")
    print(f"Lignes sans catégorie ICD-11 : {n_no_icd} "
          f"({n_no_icd/len(records)*100:.1f}%)")

    print("\nExemple de résultat (ligne avec contenu) :")
    example = next(r for r in records if r["symptoms"])
    print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
