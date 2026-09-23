# -*- coding: utf-8 -*-
"""
Détection de "small talk" : remerciements, salutations, accusés de
réception courts (« merci », « ok », « bonjour »...). Ces messages ne
décrivent aucun symptôme et ne doivent PAS être envoyés au classifieur/
NER/RAG — ça produirait une réponse absurde (comme observé : "merci
beaucoup" classé en "Mobility Problems").

Volontairement simple (mots-clés), cohérent avec l'approche déterministe
déjà utilisée pour les règles d'urgence : pas besoin d'un modèle pour
détecter un remerciement.
"""

import re

THANKS_WORDS = re.compile(
    r"\b(merci\w*|thanks?|thank you|thx|ok(ay)?|d'accord|super|parfait|cool|nice)\b",
    re.IGNORECASE,
)

GREETING_WORDS = re.compile(
    r"\b(bonjour|salut|hello|hi|coucou|bonsoir|bye|au revoir|"
    r"[aà] bient[oô]t|good bye)\b",
    re.IGNORECASE,
)

MAX_WORDS_FOR_SMALLTALK = 8  # au-delà, on considère que le message décrit probablement autre chose


def detect_smalltalk(text):
    """Retourne une réponse canned si le message est du small talk,
    sinon None (le pipeline complet doit alors s'exécuter).

    On ne se limite pas au début exact du message (« owwww merci bcp »
    doit être détecté), mais on exige que le message reste court, pour
    éviter de classer en small talk une vraie question qui contiendrait
    incidemment un mot comme "ok" au milieu d'une phrase plus longue."""
    text = text.strip()
    if not text or len(text.split()) > MAX_WORDS_FOR_SMALLTALK:
        return None

    if THANKS_WORDS.search(text):
        return "De rien, prenez bien soin de votre animal ! N'hésitez pas si vous avez d'autres questions."
    if GREETING_WORDS.search(text):
        return "Bonjour ! Décrivez-moi ce qui inquiète votre animal, je suis là pour vous aider."
    return None
