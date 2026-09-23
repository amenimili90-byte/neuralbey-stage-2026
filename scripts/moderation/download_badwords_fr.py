# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Phase 1 : corpus de mots interdits

Cahier des charges, section 4.2, "Travail du stagiaire" :
"Constitution d'un corpus de mots interdits".

Source : darwiin/french-badwords-list (GitHub, licence MIT), une liste
consolidée de mots grossiers/insultes en français, incluant de nombreuses
variantes orthographiques utilisées pour contourner les filtres
(ex. "c0n" pour "con", "3ncul3" pour "enculé"). Ces variantes sont très
utiles pour un filtre par mot-clé (le "filet de sécurité"), qui doit
attraper ces contournements que le modèle NLP pourrait manquer.

Comme pour tout corpus de modération, ce fichier contient par nature
des mots grossiers/insultants — c'est l'objet même du livrable demandé.

Sortie : processed/fr_badwords.txt (un mot/expression par ligne)

Lancer avec :
    py download_badwords_fr.py
"""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/darwiin/french-badwords-list/master/list.txt"
OUT_PATH = Path("processed/fr_badwords.txt")


def main():
    print(f"Téléchargement depuis {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content = response.read().decode("utf-8")

    words = [w.strip() for w in content.splitlines() if w.strip()]
    words = sorted(set(words))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")

    print(f" {len(words)} entrées uniques sauvegardées -> {OUT_PATH}")
    print("(Inclut des variantes orthographiques de contournement, ex. leetspeak)")


if __name__ == "__main__":
    main()
