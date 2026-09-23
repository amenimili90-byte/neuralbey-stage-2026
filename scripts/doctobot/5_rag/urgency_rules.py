# -*- coding: utf-8 -*-
"""
Phase 5 - Règles d'urgence

Cahier des charges, section 4.1 : "Gestion des cas urgents (redirection
immédiate vers un vétérinaire)".

Ces règles sont volontairement simples et déterministes (recherche de
mots-clés), PAS confiées au LLM : dans un cas de vie ou de mort pour
l'animal, on ne veut pas dépendre d'un modèle génératif qui pourrait
"oublier" de signaler l'urgence ou la minimiser. Une règle par mot-clé,
vérifiable et prévisible, passe AVANT tout le reste du pipeline.
"""

import re

# Chaque entrée : (motif regex, libellé de l'urgence en français)
# Bilingue (EN + FR) puisque les utilisateurs peuvent écrire dans les deux langues.
URGENCY_PATTERNS = [
    (r"\b(difficulty breathing|can't breathe|struggling to breathe|"
     r"gasping|choking|trouble breathing|hard time breathing|"
     r"short of breath|breathing heavily)\b", "Détresse respiratoire / étouffement"),
    (r"\b(difficult[ée]s? [aà] respirer|du mal [aà] respirer|"
     r"peine [aà] respirer|n'arrive (pas|plus) [aà] respirer|"
     r"arrive pas [aà] respirer|respire (mal|difficilement)|"
     r"s'?étouffe|étouffement|suffoque|essoufflement (important|sévère))\b",
     "Détresse respiratoire / étouffement"),

    (r"\b(seizure|seizures|convulsing|convulsion|fitting|is seizing)\b", "Crise convulsive"),
    (r"\bconvuls\w*\b", "Crise convulsive"),

    (r"\b(bloated abdomen|distended abdomen|bloat|gdv|"
     r"retching|unproductive vomiting)\b", "Suspicion de ballonnement / torsion gastrique (urgence chirurgicale)"),
    (r"\b(ventre gonfl[ée]|abdomen gonfl[ée]|ballonnement|"
     r"vomissements? improductifs?)\b", "Suspicion de ballonnement / torsion gastrique (urgence chirurgicale)"),

    (r"\b(uncontrolled bleeding|won'?t stop bleeding|heavy bleeding|"
     r"blood loss|bleeding a lot|bleeding heavily)\b", "Saignement incontrôlé"),
    (r"\b(saign[ée]?ment incontr[oô]l[ée]|saigne beaucoup|"
     r"n'arrête (pas|plus) de saigner|h[ée]morragie)\b", "Saignement incontrôlé"),

    (r"\b(collapsed|collapse|unconscious|unresponsive|not moving|"
     r"won'?t wake up|stopped moving)\b", "Effondrement / perte de conscience"),
    (r"\b(effondr[ée]?|inconscient|ne r[ée]agit plus|ne bouge plus|"
     r"s'?est effondr[ée]|ne se r[ée]veille pas)\b", "Effondrement / perte de conscience"),

    (r"\b(poison|poisoned|ingested chemical|ate chocolate|ate rat bait|"
     r"toxic)\b", "Suspicion d'intoxication"),
    (r"\b(intoxication|empoisonn[ée]|a mang[ée] du chocolat|"
     r"a ingér[ée] un produit toxique)\b", "Suspicion d'intoxication"),

    (r"\b(can'?t urinate|straining to urinate|no urine|"
     r"blocked bladder)\b", "Blocage urinaire (urgence, surtout chez le chat mâle)"),
    (r"\b(n'arrive (pas|plus) [aà] uriner|ne peut pas uriner|"
     r"bloqu[ée] urinaire)\b", "Blocage urinaire (urgence, surtout chez le chat mâle)"),

    (r"\b(hit by a car|been hit|got hit|run over|severe trauma|deep wound|"
     r"broken bone|fracture)\b", "Traumatisme sévère / fracture"),
    (r"\b(renvers[ée]?|s'?est fait renverser|accident|fracture|"
     r"plaie profonde)\b", "Traumatisme sévère / fracture"),
]

COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in URGENCY_PATTERNS]


def check_urgency(text):
    """Retourne le libellé de la première urgence détectée dans le texte,
    ou None si aucune règle ne correspond."""
    for pattern, label in COMPILED_PATTERNS:
        if pattern.search(text):
            return label
    return None


def urgency_message(label):
    """Message de redirection immédiate à afficher à l'utilisateur."""
    return (
        f" Ce que vous décrivez ({label}) peut être une urgence vétérinaire. "
        f"Merci de contacter immédiatement votre vétérinaire ou la clinique "
        f"vétérinaire d'urgence la plus proche, sans attendre. "
        f"Ce chatbot ne remplace pas un avis vétérinaire professionnel."
    )