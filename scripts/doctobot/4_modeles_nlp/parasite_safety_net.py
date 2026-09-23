"""
Doctobot - Filet de sécurité : détection de risque parasitaire par mots-clés
=================================================================================

Cahier des charges - section 4.1 : "Analyse des symptômes décrits par
l'utilisateur" doit rester fiable même dans les cas où le classifieur
CamemBERT confond "Parasites" avec une autre catégorie (cf. diagnostic
matrice de confusion : le classifieur route souvent "Parasites" vers la
catégorie du symptôme visible - digestif, cutané, auriculaire - plutôt
que vers la cause parasitaire elle-même).

Ce module ne remplace PAS le classifieur : il ajoute un signal
complémentaire indépendant, basé sur la présence de termes évoquant
explicitement un parasite, peu importe la catégorie prédite par ailleurs.
"""

import re

# Liste volontairement large : mieux vaut un faux positif (conseil
# anti-parasitaire donné à tort) qu'un faux négatif (risque parasitaire
# manqué). Inclut les variantes singulier/pluriel.
PARASITE_KEYWORDS = [
    "flea", "fleas",
    "tick", "ticks",
    "mite", "mites",
    "worm", "worms",
    "tapeworm", "roundworm", "heartworm", "hookworm",
    "maggot", "maggots", "flystrike",
    "lice", "louse",
    "mange", "scabies",
    "ringworm",  # fongique, pas parasitaire au sens strict, mais souvent
                 # confondu et pris en charge par le même type de traitement
]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in PARASITE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def detect_parasite_risk(text: str) -> bool:
    """Retourne True si le texte mentionne explicitement un terme évoquant
    un parasite, indépendamment de la catégorie prédite par le classifieur."""
    return bool(_PATTERN.search(text))


def predict_with_safety_net(text: str, classifier_predict_fn) -> dict:
    """Combine la prédiction du classifieur CamemBERT avec le filet de
    sécurité par mots-clés.

    classifier_predict_fn : fonction qui prend un texte et retourne la
    catégorie prédite (string) — à fournir par le code appelant (ex.
    la fonction d'inférence CamemBERT).
    """
    predicted_category = classifier_predict_fn(text)
    parasite_risk = detect_parasite_risk(text)

    result = {
        "predicted_category": predicted_category,
        "parasite_risk_detected": parasite_risk,
    }

    if parasite_risk and predicted_category != "Parasites":
        result["advisory"] = (
            "Un terme évoquant un parasite a été détecté dans la description, "
            f"même si la catégorie principale prédite est '{predicted_category}'. "
            "Un conseil anti-parasitaire complémentaire est ajouté par précaution."
        )

    return result
