# -*- coding: utf-8 -*-
"""
Phase 3 - Knowledge Base - Étape 3 : test de la recherche sémantique

Interroge la base Qdrant locale avec une question libre (en français ou en
anglais, le modèle est multilingue) et affiche les chunks les plus proches
sémantiquement — c'est exactement ce que fera le futur système RAG
(Phase 5) pour retrouver l'information pertinente avant de générer une
réponse.

Lancer avec :
    py query_kb.py
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "doctobot_kb"
MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 3

# Modifiez ou ajoutez des questions de test ici
TEST_QUERIES = [
    "Mon chat a des puces et se gratte beaucoup, que faire ?",
    "My dog has been vomiting for two days",
    "Quels sont les signes d'une otite chez le chien ?",
]


def main():
    print(f"Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connexion à Qdrant local ({QDRANT_PATH}/)...")
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' introuvable. "
            f"Avez-vous lancé build_embeddings_index.py ?"
        )

    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"Question : {query}")
        print(f"{'='*70}")

        query_vector = model.encode("query: " + query).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
        ).points

        for rank, hit in enumerate(results, start=1):
            print(f"\n  [{rank}] score={hit.score:.3f} "
                  f"(source : {hit.payload['source_file']}, page {hit.payload['page']})")
            preview = hit.payload["text"][:250]
            print(f"      {preview}...")


if __name__ == "__main__":
    main()
