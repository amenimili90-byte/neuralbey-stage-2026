# -*- coding: utf-8 -*-
"""
Traduit un sous-ensemble du dataset unifié (anglais) en français, en
utilisant un modèle de traduction pré-entraîné (Helsinki-NLP/opus-mt-en-fr,
disponible sur Hugging Face — même écosystème que CamemBERT).

But : donner au modèle NER des exemples de phrases FRANÇAISES pour qu'il
apprenne à reconnaître les entités dans la langue réellement utilisée par
les utilisateurs de Cheebo, et pas seulement en anglais.

Entrée  : processed/doctobot_unified_with_severity.jsonl
Sortie  : processed/doctobot_fr_translated.jsonl
          (mêmes champs que l'original + "text" remplacé par la traduction
          française + "language": "fr")

Combien de lignes traduire ? Par défaut N_TO_TRANSLATE = 3000 (~40% du
dataset). La traduction sur CPU prend du temps (comptez plusieurs bonnes
heures pour 3000 lignes) ; vous pouvez réduire ce nombre pour un premier
essai rapide (ex. 300) avant de lancer la traduction complète.

Lancer avec :
    py translate_to_french.py
"""

import json
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

RAW_PATH = Path("processed/doctobot_unified_with_severity.jsonl")
OUT_PATH = Path("processed/doctobot_fr_translated.jsonl")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

# Nombre de lignes à traduire. Réduisez à 300 pour un premier test rapide.
N_TO_TRANSLATE = 3000
BATCH_SIZE = 8


def load_records(path, limit):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("text", "").strip():
                records.append(rec)
            if len(records) >= limit:
                break
    return records


def translate_batch(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=256)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Introuvable : {RAW_PATH}")

    print(f"Chargement du modèle de traduction {MODEL_NAME}...")
    print("(premier lancement : téléchargement depuis Hugging Face, ~300 Mo)")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Chargement de {N_TO_TRANSLATE} lignes à traduire...")
    records = load_records(RAW_PATH, N_TO_TRANSLATE)
    print(f"  -> {len(records)} lignes chargées")

    translated_records = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        try:
            translations = translate_batch(texts, tokenizer, model)
        except Exception as e:
            print(f"    Erreur sur le batch {i}-{i+BATCH_SIZE} : {e}. Batch ignoré.")
            continue

        for rec, fr_text in zip(batch, translations):
            new_rec = dict(rec)
            new_rec["text"] = fr_text
            new_rec["text_original_en"] = rec["text"]  # gardé pour vérification
            new_rec["language"] = "fr"
            translated_records.append(new_rec)

        done = min(i + BATCH_SIZE, len(records))
        print(f"  Traduit : {done}/{len(records)}", end="\r")

    print(f"\n {len(translated_records)} lignes traduites")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in translated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f" Sauvegardé -> {OUT_PATH}")

    # Aperçu de quelques traductions pour vérification manuelle
    print("\n--- Exemples de traduction (à vérifier visuellement) ---")
    for rec in translated_records[:5]:
        print(f"EN: {rec['text_original_en'][:100]}")
        print(f"FR: {rec['text'][:100]}")
        print()


if __name__ == "__main__":
    main()
