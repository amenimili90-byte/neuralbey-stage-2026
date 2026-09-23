"""
Doctobot - Étape 1 : Collecte des datasets
============================================

Ce script télécharge les sources de données brutes nécessaires
au pipeline RAG de Doctobot (module vétérinaire du projet Cheebo).

Sources :
  1. Pet Health Symptoms Dataset (ouvert)  -> raw/pet_health_symptoms.csv
  2. PetEVAL / SAVSNET (accès conditionné) -> raw/peteval/  (voir README.md)

Prérequis :
    pip install datasets huggingface_hub pandas

Pour PetEVAL, tu dois d'abord :
  1. Créer un compte sur https://huggingface.co
  2. Aller sur https://huggingface.co/datasets/SAVSNET/PetEVAL
  3. Accepter les conditions d'utilisation (bouton "Agree and access repository")
  4. Générer un token d'accès : https://huggingface.co/settings/tokens
  5. Lancer `huggingface-cli login` dans ton terminal, ou définir la variable
     d'environnement HF_TOKEN avant d'exécuter ce script.
"""

import os
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)


def download_pet_health_symptoms():
    """Dataset 1 : ouvert, aucune authentification nécessaire."""
    from datasets import load_dataset

    print("Téléchargement de karenwky/pet-health-symptoms-dataset ...")
    dataset = load_dataset("karenwky/pet-health-symptoms-dataset")

    out_path = RAW_DIR / "pet_health_symptoms.csv"
    dataset["train"].to_csv(out_path)
    print(f"Sauvegardé dans {out_path} ({len(dataset['train'])} lignes)")
    print("   Colonnes attendues : text, condition, record_type")


def download_peteval():
    """Dataset 2 : nécessite un token HF avec accès accepté (voir docstring)."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("  Variable HF_TOKEN non définie.")
        print("   Assure-toi d'avoir accepté les conditions sur")
        print("   https://huggingface.co/datasets/SAVSNET/PetEVAL")
        print("   puis lance : export HF_TOKEN=ton_token")
        return

    print("Téléchargement de SAVSNET/PetEVAL ...")
    try:
        dataset = load_dataset("SAVSNET/PetEVAL", token=token)
        out_dir = RAW_DIR / "peteval"
        out_dir.mkdir(exist_ok=True)
        for split in dataset:
            dataset[split].to_csv(out_dir / f"{split}.csv")
        print(f" Sauvegardé dans {out_dir}")
    except Exception as e:
        print(f"Échec du téléchargement : {e}")
        print("   Vérifie que ton compte a bien accès au dataset (conditions acceptées).")


if __name__ == "__main__":
    download_pet_health_symptoms()
    print()
    download_peteval()
