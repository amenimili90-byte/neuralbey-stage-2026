# -*- coding: utf-8 -*-
"""
Phase 3 - Knowledge Base - Étape 2 : embeddings + indexation Qdrant

Transforme chaque chunk de texte (généré par extract_and_chunk_pdfs.py) en
vecteur numérique (embedding), et stocke ces vecteurs dans une base Qdrant
LOCALE (pas de serveur à installer, pas de Docker — les données sont
sauvegardées sur disque dans le dossier `qdrant_data/`).

Modèle d'embeddings : paraphrase-multilingual-MiniLM-L12-v2
  - Multilingue (couvre l'anglais ET le français) : cohérent avec des PDF
    qui peuvent être en anglais et des utilisateurs qui écrivent en français
  - Léger (~118M paramètres, 384 dimensions) : rapide sur CPU
  - Modèle local, téléchargé une seule fois depuis Hugging Face

Entrée : processed/kb_chunks.jsonl
Sortie : base Qdrant locale dans qdrant_data/, collection "doctobot_kb"

Lancer avec :
    py build_embeddings_index.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

CHUNKS_PATH = Path("processed/kb_chunks.jsonl")
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "doctobot_kb"
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Introuvable : {CHUNKS_PATH}. Lancez d'abord extract_and_chunk_pdfs.py"
        )

    print(f"Chargement du modèle d'embeddings {MODEL_NAME}...")
    print("(premier lancement : téléchargement depuis Hugging Face)")
    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"  -> dimension des vecteurs : {vector_size}")

    print(f"\nChargement des chunks depuis {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  -> {len(chunks)} chunks à indexer")

    print(f"\nConnexion à Qdrant local (dossier {QDRANT_PATH}/)...")
    client = QdrantClient(path=QDRANT_PATH)

    # Recrée la collection à chaque lancement (simple et sûr pour un stage :
    # pas de risque de mélanger une ancienne et une nouvelle version de l'index)
    if client.collection_exists(COLLECTION_NAME):
        print(f"  Collection '{COLLECTION_NAME}' existante -> supprimée et recréée")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print(f"\nCalcul des embeddings et indexation ({len(chunks)} chunks, "
          f"par lots de {BATCH_SIZE})...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        # Le modèle e5 attend un préfixe "passage: " devant chaque document
        # (et "query: " devant chaque question, voir query_kb.py) : c'est
        # ce qui lui permet de bien distinguer le rôle de chaque texte et
        # d'obtenir de meilleurs scores de similarité, y compris cross-langue.
        texts = ["passage: " + c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        points = [
            PointStruct(
                id=c["id"],
                vector=emb.tolist(),
                payload={
                    "text": c["text"],
                    "source_file": c["source_file"],
                    "page": c["page"],
                },
            )
            for c, emb in zip(batch, embeddings)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        done = min(i + BATCH_SIZE, len(chunks))
        print(f"  Indexé : {done}/{len(chunks)}", end="\r")

    print(f"\n\n {len(chunks)} chunks indexés dans Qdrant "
          f"(collection '{COLLECTION_NAME}', dossier {QDRANT_PATH}/)")
    print("Vous pouvez maintenant lancer query_kb.py pour tester la recherche.")


if __name__ == "__main__":
    main()
