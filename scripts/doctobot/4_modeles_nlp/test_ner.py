# -*- coding: utf-8 -*-
"""
Test rapide du modèle NER entraîné sur une ou plusieurs phrases libres.
Utile pour vérifier concrètement que l'extraction animal/symptôme/âge
fonctionne avant de le brancher dans le pipeline RAG (Phase 5).

Lancer avec :
    py test_ner.py
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

MODEL_DIR = Path("models/doctobot_ner/final")

EXAMPLES = [
    "My 4-year-old dog has been vomiting and seems very tired",
    "Mon chat de 2 ans a des puces et se gratte beaucoup",
    "My rabbit stopped eating since yesterday",
]


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier {MODEL_DIR} n'existe pas encore. "
            f"Cela veut dire que train_ner_camembert.py n'a pas terminé "
            f"(ou pas encore été lancé) avec succès. "
            f"Lancez-le d'abord, vérifiez qu'il affiche "
            f"'Modèle NER sauvegardé', puis relancez ce script."
        )
    model_path = str(MODEL_DIR.resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    for text in EXAMPLES:
        tokens = text.split()
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        word_ids = inputs.word_ids(batch_index=0)

        print(f"\nPhrase : {text}")
        seen = set()
        for idx, word_id in zip(predictions, word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            label = model.config.id2label[idx]
            if label != "O":
                print(f"  {tokens[word_id]!r:20s} -> {label}")


if __name__ == "__main__":
    main()