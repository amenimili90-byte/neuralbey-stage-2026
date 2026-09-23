"""
Doctobot - Étape 2 (partie 4) : Parseur robuste des annotations NER PetEVAL
================================================================================

`disease` et `icd_label` sont stockées comme des CHAINES DE CARACTERES dans
le dataset source (pas des listes natives). `ast.literal_eval` échoue sur
~30% des lignes de `disease`, probablement à cause d'apostrophes internes
dans le texte des entités (ex: noms de médicaments, races, etc.) qui cassent
la syntaxe Python attendue.

Ce script utilise une regex tolérante à ces apostrophes pour extraire les
entités de façon fiable, sans dépendre de literal_eval.

Usage :
    py parse_ner_column.py
"""

import ast
import json
import re
from pathlib import Path
from collections import Counter

RAW_DIR = Path(__file__).parent / "raw"
IN_PATH = RAW_DIR / "peteval_json" / "test.jsonl"
OUT_PATH = RAW_DIR / "peteval_json" / "test_cleaned.jsonl"

# Capture chaque bloc {'end': N, 'entity': '...', 'label': 'XXX', 'start': N}
# 'entity' est capturé en non-greedy jusqu'au prochain "', 'label': '<MAJUSCULES>'"
# ce qui tolère les apostrophes à l'intérieur du texte de l'entité.
ENTITY_PATTERN = re.compile(
    r"\{'end':\s*(?P<end>\d+),\s*'entity':\s*'(?P<entity>.*?)',\s*"
    r"'label':\s*'(?P<label>[A-Z]+)',\s*'start':\s*(?P<start>\d+)\}"
)


def parse_ner_string(raw: str) -> list[dict]:
    """Parse une chaîne du type "[{'end': 185, 'entity': 'otitis', ...}]"
    en liste de dicts Python, de façon tolérante aux apostrophes internes."""
    if raw is None or raw.strip() == "[]":
        return []

    matches = ENTITY_PATTERN.finditer(raw)
    results = []
    for m in matches:
        results.append({
            "start": int(m.group("start")),
            "end": int(m.group("end")),
            "entity": m.group("entity"),
            "label": m.group("label"),
        })
    return results


def parse_label_list(raw: str) -> list[str]:
    """Parse icd_label, une simple liste de chaînes sans apostrophes internes.
    ast.literal_eval suffit normalement ici."""
    if raw is None or raw.strip() == "[]":
        return []
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []


def main():
    n_total = 0
    n_disease_regex = 0
    n_disease_literal_eval_would_fail = 0
    label_counter = Counter()

    with open(IN_PATH, encoding="utf-8") as f_in, \
         open(OUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            row = json.loads(line)
            n_total += 1

            raw_disease = row.get("disease", "[]")
            raw_annon = row.get("annonymisation", "[]")
            raw_icd = row.get("icd_label", "[]")

            # Comparaison : est-ce que literal_eval aurait échoué ?
            try:
                ast.literal_eval(raw_disease)
            except (ValueError, SyntaxError):
                n_disease_literal_eval_would_fail += 1

            parsed_disease = parse_ner_string(raw_disease)
            parsed_annon = parse_ner_string(raw_annon)
            parsed_icd = parse_label_list(raw_icd)

            if parsed_disease:
                n_disease_regex += 1
                for ent in parsed_disease:
                    label_counter[ent["entity"].lower()] += 1

            cleaned_row = {
                "id": row["id"],
                "sentence": row["sentence"],
                "icd_label": parsed_icd,
                "disease": parsed_disease,
                "annonymisation": parsed_annon,
            }
            f_out.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")

    print(f"Total lignes                                    : {n_total}")
    print(f"'disease' non vide via regex robuste            : {n_disease_regex} "
          f"({n_disease_regex/n_total*100:.1f}%)")
    print(f"'disease' aurait échoué avec ast.literal_eval    : "
          f"{n_disease_literal_eval_would_fail} "
          f"({n_disease_literal_eval_would_fail/n_total*100:.1f}%)")
    print(f"\n Fichier nettoyé sauvegardé : {OUT_PATH}")

    print("\nTop 15 entités 'disease' les plus fréquentes :")
    for entity, count in label_counter.most_common(15):
        print(f"  {count:4d}  {entity}")


if __name__ == "__main__":
    main()
