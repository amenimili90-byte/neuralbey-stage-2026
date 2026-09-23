# -*- coding: utf-8 -*-
"""
Module 3 - Chatbot de navigation - Utilisation interactive

Cahier des charges, section 4.3 : "Guidage pas à pas dans l'application",
"Redirection vers les sections appropriées".

Prend une question libre, prédit l'intent, et retourne le message de
guidage + la section vers laquelle rediriger (défini dans intents_data.py).

Lancer avec :
    py navigation_chatbot.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("models/navigation_classifier/final")
META_PATH = Path("processed/navigation/intents_meta.json")

CONFIDENCE_THRESHOLD = 0.20  # calibré sur le validation set (93,8% précision / 76,2% couverture)
# Recalibrez avec calibrate_navigation_threshold.py une fois plus de données
# disponibles (val set actuel : 21 exemples seulement, résultat approximatif)


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier {MODEL_DIR} n'existe pas. Lancez d'abord "
            f"generate_training_data.py puis train_intent_classifier.py."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open(META_PATH, encoding="utf-8") as f:
        intents_meta = json.load(f)

    print(" Chatbot de navigation prêt. Tapez votre question (ou 'quit' pour arrêter).\n")

    while True:
        text = input("Vous : ").strip()
        if not text or text.lower() in ("quit", "exit"):
            break

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        intent = model.config.id2label[pred_id]
        confidence = probs[pred_id].item()

        print(f"[Debug] Intent détecté : {intent} (confiance {confidence:.1%})")

        if confidence < CONFIDENCE_THRESHOLD:
            print("Navi : Je ne suis pas sûr de bien comprendre votre demande. "
                  "Pouvez-vous reformuler, ou contacter le support ?\n")
            continue

        meta = intents_meta.get(intent, {})
        reponse = meta.get("reponse_type", "Je ne sais pas encore répondre à cela.")
        redirection = meta.get("redirection")

        print(f"Navi : {reponse}")
        if redirection:
            print(f"[Redirection app -> {redirection}]")
        print()


if __name__ == "__main__":
    main()
