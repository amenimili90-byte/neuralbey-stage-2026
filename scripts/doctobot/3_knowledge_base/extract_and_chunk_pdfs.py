# -*- coding: utf-8 -*-
"""
Phase 3 - Knowledge Base - Étape 1 : extraction + découpage des PDF

Cahier des charges, section 4.1 : le Doctobot doit pouvoir donner des
"conseils généraux et préventifs" fondés sur une base de connaissances
vétérinaire, plutôt que de tout faire deviner au modèle de langue.

Ce script :
  1. Lit tous les PDF présents dans le dossier `knowledge_base_pdfs/`
  2. Extrait le texte page par page
  3. Découpe chaque page en chunks (morceaux de texte) d'environ
     CHUNK_SIZE_WORDS mots, avec un chevauchement de OVERLAP_WORDS mots
     entre deux chunks consécutifs (pour ne pas couper une information
     importante exactement à la frontière de deux chunks)
  4. Sauvegarde le résultat dans processed/kb_chunks.jsonl

Placez vos PDF vétérinaires dans un dossier `knowledge_base_pdfs/` à côté
de ce script avant de le lancer.

Lancer avec :
    py extract_and_chunk_pdfs.py
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path("knowledge_base_pdfs")
OUT_PATH = Path("processed/kb_chunks.jsonl")

CHUNK_SIZE_WORDS = 220
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 40  # chunks plus courts = souvent des pages de titre/couverture


def is_boilerplate(text):
    """Détecte les chunks de type page de titre ou sommaire (table des
    matières), qui n'apportent aucune information utile mais obtiennent
    parfois un score de similarité trompeusement élevé car ils sont courts
    et génériques."""
    n_words = len(text.split())
    if n_words < MIN_CHUNK_WORDS:
        return True
    # Sommaire typique : beaucoup de points de suite "......" (dot leaders)
    # ou de numéros de page collés au texte
    dot_leader_count = text.count("....")
    if dot_leader_count >= 2:
        return True
    return False


def clean_text(text):
    """Nettoyage léger : espaces multiples, retours à la ligne parasites."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS):
    """Découpe un texte en chunks de `chunk_size` mots, avec `overlap`
    mots de chevauchement entre deux chunks consécutifs."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def extract_pdf(pdf_path):
    """Retourne une liste de dicts {page, text} pour un PDF donné."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"    Erreur extraction page {i+1} de {pdf_path.name} : {e}")
            text = ""
        text = clean_text(text)
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def main():
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier {PDF_DIR} n'existe pas. Créez-le et placez-y vos "
            f"PDF vétérinaires avant de relancer ce script."
        )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun fichier .pdf trouvé dans {PDF_DIR}.")

    print(f"{len(pdf_files)} PDF trouvé(s) : {[f.name for f in pdf_files]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    chunk_id = 0

    for pdf_path in pdf_files:
        print(f"\nTraitement de {pdf_path.name}...")
        pages = extract_pdf(pdf_path)
        print(f"  {len(pages)} pages avec texte extrait")

        for page_info in pages:
            chunks = chunk_text(page_info["text"])
            for chunk in chunks:
                if is_boilerplate(chunk):
                    continue
                all_chunks.append({
                    "id": chunk_id,
                    "source_file": pdf_path.name,
                    "page": page_info["page"],
                    "text": chunk,
                    "n_words": len(chunk.split()),
                })
                chunk_id += 1

    print(f"\n Total : {len(all_chunks)} chunks générés depuis {len(pdf_files)} PDF")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f" Sauvegardé -> {OUT_PATH}")

    if all_chunks:
        print("\n--- Exemple de chunk (le premier) ---")
        print(f"Source : {all_chunks[0]['source_file']}, page {all_chunks[0]['page']}")
        print(all_chunks[0]["text"][:300] + "...")


if __name__ == "__main__":
    main()