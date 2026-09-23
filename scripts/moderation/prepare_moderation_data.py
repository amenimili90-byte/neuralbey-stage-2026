# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Phase 2 : préparation finale du dataset

Combine processed/moderation_en_raw.jsonl (anglais, complet) et
processed/moderation_fr_translated.jsonl (français, sous-ensemble traduit),
puis découpe en train/val/test de façon STRATIFIÉE (même proportion de
chaque classe dans les 3 splits) — important ici car la classe "À bloquer"
est minoritaire (5,8%) : un split non stratifié pourrait, par malchance,
en priver complètement le test set.

Sortie : processed/moderation/{train,val,test}.jsonl

Lancer avec :
    py prepare_moderation_data.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

EN_PATH = Path("processed/moderation_en_raw.jsonl")
FR_PATH = Path("processed/moderation_fr_translated.jsonl")
OUT_DIR = Path("processed/moderation")

random.seed(42)


def load_jsonl(path):
    if not path.exists():
        print(f"  (fichier {path} introuvable, ignoré)")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(records, train_ratio=0.8, val_ratio=0.1):
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, val, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def print_distribution(name, records):
    counts = defaultdict(int)
    lang_counts = defaultdict(int)
    for r in records:
        counts[r["label"]] += 1
        lang_counts[r.get("language", "en")] += 1
    total = len(records)
    dist_str = ", ".join(f"{k}: {v} ({100*v/total:.1f}%)" for k, v in counts.items())
    lang_str = ", ".join(f"{k}: {v}" for k, v in lang_counts.items())
    print(f"  {name:6s} ({total} lignes) -> {dist_str} | langues : {lang_str}")


def main():
    print("Chargement des données anglaises...")
    en_records = load_jsonl(EN_PATH)
    print(f"  -> {len(en_records)} lignes")

    print("Chargement des données françaises traduites...")
    fr_records = load_jsonl(FR_PATH)
    print(f"  -> {len(fr_records)} lignes")

    all_records = en_records + fr_records
    print(f"\nTotal combiné : {len(all_records)} lignes")

    train, val, test = stratified_split(all_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = OUT_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nRépartition finale (vérifiez que chaque classe est bien représentée "
          "dans les 3 splits) :")
    print_distribution("train", train)
    print_distribution("val", val)
    print_distribution("test", test)


if __name__ == "__main__":
    main()
