# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Phase 1 : collecte des données

Cahier des charges, section 4.2 : classification en 3 niveaux
(Acceptable / À modérer / À bloquer).

Source : tdavidson/hate_speech_offensive (Hugging Face), 24 783 tweets
annotés par plusieurs annotateurs humains. Référence académique standard
pour ce type de tâche (Davidson et al., ICWSM 2017).

 Ce dataset contient, par nature, des propos racistes/sexistes/
homophobes et autres contenus offensants — c'est indispensable pour
entraîner un modèle à les reconnaître, exactement comme un antivirus a
besoin d'exemples de virus. Le contenu n'est manipulé ici qu'à des fins
de modération/sécurité.

Mapping retenu vers les 3 catégories du cahier des charges :
  - classe "neither" (2)            -> Acceptable
  - classe "offensive language" (1) -> À modérer
  - classe "hate speech" (0)        -> À bloquer

Sortie : processed/moderation_en_raw.jsonl

Lancer avec :
    py download_moderation_data.py
"""

import json
from pathlib import Path
from datasets import load_dataset

OUT_PATH = Path("processed/moderation_en_raw.jsonl")

LABEL_MAP = {
    0: "À bloquer",     # hate speech
    1: "À modérer",     # offensive language
    2: "Acceptable",    # neither
}


def main():
    print("Téléchargement du dataset tdavidson/hate_speech_offensive...")
    ds = load_dataset("tdavidson/hate_speech_offensive", split="train")
    print(f"  -> {len(ds)} exemples chargés")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    counts = {"Acceptable": 0, "À modérer": 0, "À bloquer": 0}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            label = LABEL_MAP[row["class"]]
            counts[label] += 1
            record = {
                "text": row["tweet"],
                "label": label,
                "language": "en",
                "source": "tdavidson/hate_speech_offensive",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n Sauvegardé -> {OUT_PATH}")
    print("Répartition des classes :")
    for label, count in counts.items():
        pct = 100 * count / len(ds)
        print(f"  {label:12s}: {count:6d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
