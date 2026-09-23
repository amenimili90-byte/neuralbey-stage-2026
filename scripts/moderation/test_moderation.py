# -*- coding: utf-8 -*-
"""
Module 2 - Modération de contenu - Test sur phrases françaises

Teste le modèle entraîné sur des messages français variés, écrits à la
main, pour observer concrètement son comportement (notamment le taux de
faux positifs sur "À bloquer" identifié sur le test set automatique).

Lancer avec :
    py test_moderation.py
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from moderation_correction import decide_label

MODEL_DIR = Path("models/moderation_classifier/final")

# (texte, catégorie attendue) - catégorie attendue = ce qu'un humain dirait
TEST_MESSAGES = [
    # --- Acceptable ---
    ("Merci pour tous ces conseils, mon chien se sent beaucoup mieux !", "Acceptable"),
    ("Est-ce que quelqu'un connaît un bon vétérinaire à Sousse ?", "Acceptable"),
    ("Je ne suis pas d'accord avec ton avis mais je respecte ton choix.", "Acceptable"),
    ("Mon chat est adorable, il dort toute la journée.", "Acceptable"),
    ("Franchement je trouve que cette application est mal faite.", "Acceptable"),

    # --- À modérer (grossier/insultant mais pas une menace) ---
    ("Tu es vraiment stupide de faire ça à ton animal.", "À modérer"),
    ("Ferme-la, personne ne t'a demandé ton avis.", "À modérer"),
    ("C'est un idiot celui qui a écrit ce commentaire.", "À modérer"),
    ("Va te faire voir avec tes conseils à deux balles.", "À modérer"),
    ("T'es qu'un abruti, tu comprends jamais rien.", "À modérer"),

    # --- À bloquer (menace, insulte grave, harcèlement) ---
    ("Je vais te retrouver et te faire du mal, sale ordure.", "À bloquer"),
    ("Tu ne mérites pas de vivre, disparais de cette communauté.", "À bloquer"),
    ("Arrête de me suivre et de me harceler sur chaque post ou tu vas le regretter.", "À bloquer"),
    ("Je vais tout faire pour détruire ta vie, tu vas payer.", "À bloquer"),
    ("Continue comme ça et tu verras ce qui va t'arriver, je te le promets.", "À bloquer"),
]


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Introuvable : {MODEL_DIR}. Avez-vous bien lancé "
                                 f"train_moderation_classifier.py ?")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    n_correct_raw = 0
    n_correct_corrected = 0
    print(f"{'Message':45s} {'Attendu':11s} {'Brut':11s} {'Corrigé':11s} {'Conf.':>7s}")
    print("-" * 100)

    for text, expected in TEST_MESSAGES:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs_tensor = torch.softmax(logits, dim=-1)[0]
        probs = {model.config.id2label[i]: probs_tensor[i].item() for i in range(len(probs_tensor))}
        pred_id = int(torch.argmax(probs_tensor))
        pred_label = model.config.id2label[pred_id]
        confidence = probs_tensor[pred_id].item()

        corrected_label, reason = decide_label(text, probs)

        raw_ok = pred_label == expected
        corrected_ok = corrected_label == expected
        n_correct_raw += raw_ok
        n_correct_corrected += corrected_ok

        status = "✅" if corrected_ok else "❌"
        short_text = text[:42] + "..." if len(text) > 42 else text
        print(f"{status} {short_text:43s} {expected:11s} {pred_label:11s} "
              f"{corrected_label:11s} {confidence:>6.1%}")

    print("-" * 100)
    print(f"\nAvant correction : {n_correct_raw}/{len(TEST_MESSAGES)} corrects")
    print(f"Après correction  : {n_correct_corrected}/{len(TEST_MESSAGES)} corrects")


if __name__ == "__main__":
    main()
