# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Phase 1/2 : traduction française

Contrairement à la première tentative sur le NER (où on avait pris les
premières lignes du dataset sans réfléchir à l'équilibre des classes,
et où on avait dû corriger après coup), on échantillonne ici de façon
STRATIFIÉE : le même nombre d'exemples par classe, pour ne pas aggraver
le déséquilibre déjà présent dans les données anglaises (77% "À modérer",
seulement 6% "À bloquer").

Modèle de traduction : Helsinki-NLP/opus-mt-en-fr (le même que pour le NER).

Entrée  : processed/moderation_en_raw.jsonl
Sortie  : processed/moderation_fr_translated.jsonl

Lancer avec :
    py translate_moderation_to_french.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

RAW_PATH = Path("processed/moderation_en_raw.jsonl")
OUT_PATH = Path("processed/moderation_fr_translated.jsonl")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
N_PER_CLASS = 1500  # échantillon équilibré : jusqu'à 1500 exemples par classe
BATCH_SIZE = 8

random.seed(42)


def load_and_stratify(path, n_per_class):
    by_label = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_label[rec["label"]].append(rec)

    sampled = []
    for label, records in by_label.items():
        random.shuffle(records)
        take = records[:n_per_class]
        sampled.extend(take)
        print(f"  {label:12s}: {len(take)} exemples sélectionnés "
              f"(sur {len(records)} disponibles)")
    random.shuffle(sampled)
    return sampled


def translate_batch(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=128)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Introuvable : {RAW_PATH}. Lancez d'abord download_moderation_data.py")

    print("Échantillonnage stratifié par classe :")
    records = load_and_stratify(RAW_PATH, N_PER_CLASS)
    print(f"\nTotal à traduire : {len(records)} exemples")

    print(f"\nChargement du modèle de traduction {MODEL_NAME}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()

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
            new_rec["text_original_en"] = rec["text"]
            new_rec["language"] = "fr"
            translated_records.append(new_rec)

        done = min(i + BATCH_SIZE, len(records))
        print(f"  Traduit : {done}/{len(records)}", end="\r")

    print(f"\n {len(translated_records)} lignes traduites")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in translated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f" Sauvegardé -> {OUT_PATH}")

    print("\n--- Exemples de traduction (à vérifier visuellement) ---")
    for rec in translated_records[:5]:
        print(f"[{rec['label']}] EN: {rec['text_original_en'][:90]}")
        print(f"          FR: {rec['text'][:90]}")
        print()


if __name__ == "__main__":
    main()
