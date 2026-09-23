# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Décision finale (seuil calibré + escalade)

Historique de conception (important pour le rapport de stage) :
  1. Première tentative : règle "pas de menace explicite -> rétrograder
     À bloquer". Fonctionnait à 15/15 sur des exemples écrits à la main,
     mais catastrophique sur le vrai test set (recall À bloquer : 72% -> 0%).
     Cause : le vrai contenu haineux du dataset est fait d'insultes/propos
     discriminatoires, pas de menaces explicites comme dans nos exemples.
  2. Version corrigée (celle-ci) : seuil de probabilité calibré sur le
     validation set (cahier des charges : "mise en place de seuils de
     confiance"), pas une règle de mots-clés inventée. Seuil retenu : 0.70
     (43% précision / 68% rappel sur "À bloquer", mesuré sur le
     validation set) — un compromis choisi consciemment, pas optimal sur
     le papier (F1) mais jugé plus raisonnable pour un premier système.

Les règles d'escalade (menace explicite, expressions vulgaires composées)
sont conservées : elles n'ont jamais causé de dégradation, puisqu'elles
ne font qu'AJOUTER des détections, jamais en retirer.
"""

import re
from pathlib import Path

BADWORDS_PATH = Path("processed/fr_badwords.txt")

THRESHOLD_A_BLOQUER = 0.70

THREAT_PATTERNS = re.compile(
    r"\b(je vais te|tu vas payer|te faire du mal|te d[ée]truire|"
    r"te retrouver|disparais|tu ne m[ée]rites pas de vivre|"
    r"te tuer|te frapper|te faire souffrir|harceler|te suivre partout|"
    r"tu verras ce qui va t'?arriver|gare [aà] toi|tu vas le regretter)\b",
    re.IGNORECASE,
)

VULGAR_PHRASES = [
    "va te faire voir", "va te faire foutre", "je t'emmerde",
    "ferme ta gueule", "ta gueule", "nique ta mère", "sale connard",
    "à deux balles", "ferme-la", "ferme la", "la ferme", "tais-toi",
    "tais toi", "occupe-toi de tes affaires", "mêle-toi de tes affaires",
]


def load_badwords(path=BADWORDS_PATH):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip()}


BADWORDS = load_badwords()


def contains_badword(text, badwords=BADWORDS):
    words = re.findall(r"\w+", text.lower())
    return any(w in badwords for w in words)


def contains_vulgar_phrase(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in VULGAR_PHRASES)


def contains_threat(text):
    return bool(THREAT_PATTERNS.search(text))


def decide_label(text, probs, threshold=THRESHOLD_A_BLOQUER):
    """Détermine la catégorie finale à partir des probabilités du modèle
    (dict {"Acceptable": p, "À modérer": p, "À bloquer": p}) et du texte.

    Ordre de décision :
      1. Menace explicite détectée -> "À bloquer" (priorité absolue,
         quelle que soit la probabilité du modèle)
      2. Probabilité de "À bloquer" >= seuil calibré -> "À bloquer"
      3. Sinon, comparaison entre "Acceptable" et "À modérer" uniquement
      4. Si le résultat est "Acceptable" mais qu'une expression vulgaire
         ou un mot du corpus interdits est détecté -> "À modérer"

    Retourne (label, raison)."""
    if contains_threat(text):
        return "À bloquer", "menace explicite détectée"

    if probs["À bloquer"] >= threshold:
        return "À bloquer", f"probabilité {probs['À bloquer']:.2f} >= seuil {threshold}"

    label = "Acceptable" if probs["Acceptable"] >= probs["À modérer"] else "À modérer"

    if label == "Acceptable" and (contains_vulgar_phrase(text) or contains_badword(text)):
        return "À modérer", "escaladé : expression vulgaire/mot interdit détecté"

    return label, f"probabilité {probs['À bloquer']:.2f} < seuil {threshold}"
