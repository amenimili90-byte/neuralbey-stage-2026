# -*- coding: utf-8 -*-
"""
Test rapide et isolé des règles d'urgence (urgency_rules.py), sans charger
aucun modèle lourd — utile pour vérifier une correction en quelques
secondes avant de relancer tout le pipeline RAG.

Lancer avec :
    py test_urgency_rules.py
"""

from urgency_rules import check_urgency

TEST_PHRASES = [
    # Cas qui DOIVENT déclencher une urgence
    ("Mon chien de 5 ans a du mal à respirer", True),
    ("My dog is having trouble breathing", True),
    ("Mon chat convulse depuis 2 minutes", True),
    ("My cat is having a seizure", True),
    ("Il ne bouge plus et ne réagit pas", True),
    ("My dog was hit by a car", True),
    ("Mon chien s'est fait renverser par une voiture", True),
    ("Il n'arrête pas de saigner depuis la patte", True),
    ("Mon chat a le ventre gonflé et essaie de vomir sans rien sortir", True),
    ("Il n'arrive plus à uriner depuis ce matin", True),

    # Cas qui ne doivent PAS déclencher d'urgence (cas bénins)
    ("Mon chat a des puces et se gratte beaucoup", False),
    ("Mon chien mange moins depuis hier", False),
    ("My cat has been sneezing a little", False),
]


def main():
    n_correct = 0
    for phrase, expected_urgent in TEST_PHRASES:
        result = check_urgency(phrase)
        is_urgent = result is not None
        status = "✅" if is_urgent == expected_urgent else "❌"
        if is_urgent == expected_urgent:
            n_correct += 1
        print(f"{status} [{'urgence attendue' if expected_urgent else 'pas d urgence':17s}] "
              f"{phrase!r:60s} -> {result}")

    print(f"\n{n_correct}/{len(TEST_PHRASES)} tests corrects")


if __name__ == "__main__":
    main()
