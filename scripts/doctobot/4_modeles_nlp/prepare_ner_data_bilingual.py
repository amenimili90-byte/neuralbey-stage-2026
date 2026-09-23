# -*- coding: utf-8 -*-
"""
Version bilingue de prepare_ner_data.py : génère le dataset BIO en combinant
- les lignes originales en ANGLAIS (dataset complet, comme avant)
- les lignes TRADUITES en FRANÇAIS (sous-ensemble produit par
  translate_to_french.py), étiquetées avec les dictionnaires français
  (fr_dictionaries.py)

Objectif : que le modèle NER voie des exemples dans les deux langues et
apprenne à reconnaître les entités quelle que soit la langue du message
utilisateur (crucial puisque les utilisateurs de Cheebo écrivent en
français).

Entrée :
  processed/doctobot_unified_with_severity.jsonl   (anglais, complet)
  processed/doctobot_fr_translated.jsonl           (français, sous-ensemble)
Sortie :
  processed/ner/{train,val,test}.jsonl  (écrase l'ancienne version anglais
  uniquement — une sauvegarde de l'ancienne version est faite automatiquement)

Lancer avec :
    py prepare_ner_data_bilingual.py
"""

import json
import re
import random
import shutil
from pathlib import Path

from fr_dictionaries import SPECIES_FR, SYMPTOMS_FR

RAW_EN_PATH = Path("processed/doctobot_unified_with_severity.jsonl")
RAW_FR_PATH = Path("processed/doctobot_fr_translated.jsonl")
OUT_DIR = Path("processed/ner")

random.seed(42)

AGE_PATTERNS = [
    r"\b\d+[\s-]year[\s-]old\b",
    r"\b\d+\s+years?\s+old\b",
    r"\b\d+\s*ans?\b",
    r"\bâgé[e]?\s+de\s+\d+\s*ans?\b",
]
AGE_REGEX = re.compile("|".join(AGE_PATTERNS), re.IGNORECASE)


def find_span(text_lower, phrase):
    if not phrase:
        return None
    phrase = phrase.strip().lower()
    if not phrase:
        return None
    pattern = r"\b" + re.escape(phrase) + r"\b"
    m = re.search(pattern, text_lower)
    if m:
        return m.start(), m.end()
    return None


def tag_record(record, lang="en"):
    text = record.get("text") or ""
    if not text.strip():
        return None
    text_lower = text.lower()

    spans = []

    species = record.get("species")
    if species and species != "unknown":
        species_key = species.lower().strip()
        if lang == "fr":
            phrase = SPECIES_FR.get(species_key)
        else:
            phrase = species_key
        if phrase:
            span = find_span(text_lower, phrase)
            if span:
                spans.append((span[0], span[1], "SPECIES"))

    for symptom in record.get("symptoms", []) or []:
        symptom_key = symptom.lower().strip()
        if lang == "fr":
            phrase = SYMPTOMS_FR.get(symptom_key)
        else:
            phrase = symptom_key
        if phrase:
            span = find_span(text_lower, phrase)
            if span:
                spans.append((span[0], span[1], "SYMPTOM"))

    for m in AGE_REGEX.finditer(text):
        spans.append((m.start(), m.end(), "AGE"))

    if not spans:
        return None

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    clean_spans = []
    last_end = -1
    for start, end, label in spans:
        if start >= last_end:
            clean_spans.append((start, end, label))
            last_end = end

    tokens = []
    labels = []
    for tok_match in re.finditer(r"\S+", text):
        tok_start, tok_end = tok_match.start(), tok_match.end()
        tok_text = tok_match.group()
        label = "O"
        for start, end, ent_label in clean_spans:
            if tok_start >= start and tok_end <= end:
                label = ("B-" if tok_start == start else "I-") + ent_label
                break
            elif tok_start < end and tok_end > start:
                label = ("B-" if tok_start <= start else "I-") + ent_label
                break
        tokens.append(tok_text)
        labels.append(label)

    if all(l == "O" for l in labels):
        return None

    return {"id": record.get("id"), "source": record.get("source"), "language": lang,
            "tokens": tokens, "ner_tags": labels}


def load_and_tag(path, lang):
    kept = []
    total = 0
    if not path.exists():
        print(f"  (fichier {path} introuvable, ignoré)")
        return kept, total
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            tagged = tag_record(record, lang=lang)
            if tagged:
                kept.append(tagged)
    return kept, total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarde de l'ancien dataset anglais-only, si présent
    for name in ["train", "val", "test"]:
        old_path = OUT_DIR / f"{name}.jsonl"
        if old_path.exists():
            backup_path = OUT_DIR / f"{name}_en_only_backup.jsonl"
            shutil.copy(old_path, backup_path)
    print("(ancien dataset anglais-only sauvegardé en *_en_only_backup.jsonl)")

    print("\n--- Traitement anglais ---")
    kept_en, total_en = load_and_tag(RAW_EN_PATH, "en")
    print(f"Lignes anglaises lues : {total_en}, conservées : {kept_en and len(kept_en)} "
          f"({100*len(kept_en)/total_en:.1f}%)" if total_en else "Aucune donnée anglaise")

    print("\n--- Traitement français (traduit) ---")
    kept_fr, total_fr = load_and_tag(RAW_FR_PATH, "fr")
    if total_fr:
        print(f"Lignes françaises lues : {total_fr}, conservées : {len(kept_fr)} "
              f"({100*len(kept_fr)/total_fr:.1f}%)")
    else:
        print("  Aucune donnée française trouvée. Avez-vous lancé translate_to_french.py ?")

    all_kept = kept_en + kept_fr
    print(f"\nTotal combiné : {len(all_kept)} lignes "
          f"({len(kept_en)} EN + {len(kept_fr)} FR)")

    counts = {"SPECIES": 0, "SYMPTOM": 0, "AGE": 0}
    counts_by_lang = {"en": dict(counts), "fr": dict(counts)}
    for rec in all_kept:
        lang = rec["language"]
        for lab in rec["ner_tags"]:
            if lab.startswith("B-"):
                counts[lab[2:]] += 1
                counts_by_lang[lang][lab[2:]] += 1

    print("--- Entités détectées (total) ---")
    for k, v in counts.items():
        print(f"  {k:8s}: {v}  (EN: {counts_by_lang['en'][k]}, FR: {counts_by_lang['fr'][k]})")

    random.shuffle(all_kept)
    n = len(all_kept)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = all_kept[:n_train]
    val = all_kept[n_train:n_train + n_val]
    test = all_kept[n_train + n_val:]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = OUT_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in split:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_fr_split = sum(1 for r in split if r["language"] == "fr")
        print(f"{name:5s} : {len(split)} lignes ({n_fr_split} FR) -> {out_path}")


if __name__ == "__main__":
    main()
