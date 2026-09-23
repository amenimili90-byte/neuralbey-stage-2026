# -*- coding: utf-8 -*-
"""
Phase 5 - Pipeline RAG complet

Orchestre tous les composants construits dans les phases précédentes :
  1. Règles d'urgence (Phase 5, urgency_rules.py) - vérifiées EN PREMIER
  2. Classifieur CamemBERT (Phase 4) -> catégorie médicale
  3. NER CamemBERT (Phase 4, bilingue) -> animal / symptôme(s) / âge
  4. Qdrant + embeddings e5 (Phase 3) -> passages pertinents des PDF
  5. LLM local (Phase 5) -> génère la réponse finale à partir de tout le contexte

Modèle LLM choisi : Qwen2.5-1.5B-Instruct, au format GGUF (quantisé),
exécuté via llama-cpp-python. Ce choix respecte la contrainte "modèle
local" (confirmée par l'encadrant) : pas d'appel API externe, le modèle
tourne entièrement sur votre machine, y compris sans GPU. C'est un modèle
volontairement petit (1.5 milliard de paramètres) pour rester utilisable
sur CPU ; il gère correctement le français et l'anglais.

Le modèle est téléchargé automatiquement depuis Hugging Face au premier
lancement (~1 Go).

Lancer avec :
    py rag_pipeline.py
Puis tapez vos questions (message vide ou "quit" pour arrêter).
"""

import sys
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from urgency_rules import check_urgency, urgency_message
from vets_lookup import find_vets, format_vet_line
from smalltalk import detect_smalltalk

# ============================================================
# CONFIGURATION - à vérifier/ajuster selon vos chemins réels
# ============================================================
CLASSIFIER_DIR = "models/camembert_condition_classifier/final"
NER_DIR = "models/doctobot_ner/final"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "doctobot_kb"
TOP_K_CHUNKS = 3

# LLM local via Transformers (pas de compilateur C++ nécessaire, contrairement
# à llama-cpp-python). Qwen2.5-1.5B-Instruct : petit, multilingue correct,
# gérable sur CPU pour une génération à la fois (comptez plusieurs secondes
# à ~1 minute par réponse selon la machine, ce qui reste acceptable pour un
# usage interactif, contrairement à un entraînement).
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


SYSTEM_PROMPT = """Tu es Doctobot, un assistant conversationnel pour les propriétaires \
d'animaux de compagnie sur la plateforme Cheebo. Tu réponds en français, de façon \
empathique, claire et concise (5-8 phrases maximum).

Règles strictes :
- Tu ne remplaces JAMAIS un vétérinaire. Tu donnes uniquement des conseils \
généraux et préventifs.
- Si les informations fournies (catégorie, entités, passages de référence) \
suggèrent un cas grave ou incertain, recommande explicitement de consulter \
un vétérinaire.
- Appuie-toi sur les passages de référence fournis quand ils sont pertinents, \
mais ne les cite pas mot pour mot : reformule avec tes propres mots.
- Si les passages de référence ne couvrent pas le sujet, dis-le honnêtement \
plutôt que d'inventer une information médicale.
"""


def load_classifier(model_dir):
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Le dossier {model_dir} n'existe pas. Vérifiez CLASSIFIER_DIR en "
            f"haut de ce script et corrigez-le pour qu'il pointe vers le "
            f"dossier où le classifieur a été sauvegardé (celui contenant "
            f"config.json et tokenizer_config.json)."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def load_ner(model_dir):
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Le dossier {model_dir} n'existe pas. Vérifiez NER_DIR en haut "
            f"de ce script."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def classify_category(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1)[0])
    label = model.config.id2label.get(pred_id, str(pred_id))
    return label


def extract_entities(text, tokenizer, model):
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = inputs.word_ids(batch_index=0)

    entities = {"SPECIES": [], "SYMPTOM": [], "AGE": []}
    seen = set()
    for idx, word_id in zip(predictions, word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        label = model.config.id2label[idx]
        if label != "O":
            ent_type = label.split("-")[-1]
            entities.setdefault(ent_type, []).append(tokens[word_id])
    return entities


def retrieve_kb_chunks(text, embed_model, qdrant_client, top_k=TOP_K_CHUNKS):
    query_vector = embed_model.encode("query: " + text).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    return [
        {"text": r.payload["text"], "source": r.payload["source_file"], "score": r.score}
        for r in results
    ]


def build_prompt(user_text, category, entities, chunks):
    entities_str = ", ".join(
        f"{k}: {', '.join(v)}" for k, v in entities.items() if v
    ) or "aucune entité détectée"

    chunks_str = "\n---\n".join(c["text"][:500] for c in chunks) if chunks else "(aucun passage pertinent trouvé)"

    return (
        f"Message de l'utilisateur : {user_text}\n\n"
        f"Catégorie médicale détectée : {category}\n"
        f"Entités extraites : {entities_str}\n\n"
        f"Passages de référence (base de connaissances vétérinaire) :\n{chunks_str}\n\n"
        f"Réponds maintenant au propriétaire de l'animal."
    )


def main():
    print("Chargement du classifieur...")
    clf_tokenizer, clf_model = load_classifier(CLASSIFIER_DIR)

    print("Chargement du NER...")
    ner_tokenizer, ner_model = load_ner(NER_DIR)

    print("Chargement du modèle d'embeddings...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connexion à Qdrant...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    print("Chargement du LLM local (Qwen2.5-1.5B-Instruct, via Transformers)...")
    print("(premier lancement : téléchargement depuis Hugging Face, ~3 Go)")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float32)
    llm_model.eval()

    print("\n Pipeline prêt. Tapez votre question (ou 'quit' pour arrêter).\n")

    while True:
        user_text = input("Vous : ").strip()
        if not user_text or user_text.lower() in ("quit", "exit"):
            break

        # --- Étape 0 : small talk (remerciements, salutations) ---
        smalltalk_reply = detect_smalltalk(user_text)
        if smalltalk_reply:
            print(f"\nDoctobot : {smalltalk_reply}\n")
            continue

        # --- Étape 1 : règles d'urgence (priorité absolue) ---
        urgency = check_urgency(user_text)
        if urgency:
            print(f"\nDoctobot : {urgency_message(urgency)}\n")
            location = input(
                "Doctobot : Dans quelle ville ou quel gouvernorat êtes-vous, "
                "pour que je vous indique le vétérinaire le plus proche ?\nVous : "
            ).strip()
            if location:
                vets, governorate = find_vets(location)
                if vets:
                    gov_label = governorate or "votre région"
                    print(f"\nDoctobot : Voici des vétérinaires disponibles ({gov_label}) :")
                    for v in vets:
                        print(f"  - {format_vet_line(v)}")
                    print()
                else:
                    print("\nDoctobot : Je n'ai pas trouvé de vétérinaire correspondant "
                          "dans ma base pour l'instant. Contactez le vétérinaire le plus "
                          "proche de vous ou les urgences vétérinaires de votre région.\n")
            continue

        # --- Étape 2 : classification ---
        category = classify_category(user_text, clf_tokenizer, clf_model)

        # --- Étape 3 : extraction d'entités ---
        entities = extract_entities(user_text, ner_tokenizer, ner_model)

        # --- Étape 4 : recherche dans la Knowledge Base ---
        chunks = retrieve_kb_chunks(user_text, embed_model, qdrant_client)

        # --- Étape 5 : génération de la réponse par le LLM ---
        prompt = build_prompt(user_text, category, entities, chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_input = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = llm_tokenizer(chat_input, return_tensors="pt")
        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = llm_tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\n[Debug] Catégorie : {category} | Entités : {entities}")
        print(f"[Debug] Chunks KB : {[c['source'] for c in chunks]}")
        print(f"\nDoctobot : {answer}\n")


if __name__ == "__main__":
    main()
