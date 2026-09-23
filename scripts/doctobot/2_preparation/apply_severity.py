"""
Doctobot - Phase 2 (fin) : Calcul de la gravité (severity) sur le dataset unifié
====================================================================================

Usage :
    py apply_severity.py
"""

import json
from pathlib import Path
from collections import Counter

from severity_rules import compute_severity

PROCESSED_DIR = Path(__file__).parent / "processed"
IN_PATH = PROCESSED_DIR / "doctobot_unified.jsonl"
OUT_PATH = PROCESSED_DIR / "doctobot_unified_with_severity.jsonl"


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    for r in records:
        r["severity"] = compute_severity(r)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" {len(records)} enregistrements -> {OUT_PATH}")

    counts = Counter(r["severity"] for r in records)
    print("\n--- Répartition des niveaux de gravité ---")
    for level in ["high", "medium", "low", "unknown"]:
        n = counts.get(level, 0)
        print(f"  {level:8s} : {n:5d} ({n/len(records)*100:.1f}%)")

    print("\n--- Exemples 'high' ---")
    for r in [r for r in records if r["severity"] == "high"][:3]:
        print(f"  [{r['source']}] {r['text'][:100]}")
        print(f"    symptoms={r['symptoms']}  disease_label={r['disease_label']}")


if __name__ == "__main__":
    main()
