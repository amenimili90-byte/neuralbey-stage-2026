"""
Doctobot - Dictionnaires de normalisation
=============================================

Utilisé par tous les scripts de transformation (transform_*.py) pour
garantir un vocabulaire unique à travers les 3 sources de données.
"""

import re

# ---------------------------------------------------------------------------
# Normalisation des espèces
# ---------------------------------------------------------------------------
SPECIES_MAP = {
    "dog": "dog", "dogs": "dog", "canine": "dog", "puppy": "dog", "puppies": "dog",
    "cat": "cat", "cats": "cat", "feline": "cat", "kitten": "cat", "kittens": "cat",
    "rabbit": "rabbit", "rabbits": "rabbit", "bunny": "rabbit",
    "cow": "cow", "cows": "cow", "cattle": "cow", "bovine": "cow", "calf": "cow",
    "goat": "goat", "goats": "goat", "caprine": "goat",
    "horse": "horse", "horses": "horse", "equine": "horse", "foal": "horse",
    "pig": "pig", "pigs": "pig", "porcine": "pig", "swine": "pig", "piglet": "pig",
    "sheep": "sheep", "ovine": "sheep", "lamb": "sheep",
    "guinea pig": "guinea_pig", "guinea pigs": "guinea_pig",
    "hamster": "hamster", "hamsters": "hamster",
    "ferret": "ferret", "ferrets": "ferret",
    "bird": "bird", "birds": "bird", "parrot": "bird", "budgie": "bird",
    "cockatiel": "bird", "canary": "bird", "finch": "bird", "lovebird": "bird",
    "parakeet": "bird", "cockatoo": "bird",
    "turtle": "reptile", "tortoise": "reptile", "lizard": "reptile", "snake": "reptile",
    "fish": "fish",
}

# Ordre important : les mots les plus longs d'abord pour éviter les faux
# positifs (ex. "cattle" avant "cat" si on faisait une recherche naïve)
_SPECIES_KEYWORDS_SORTED = sorted(SPECIES_MAP.keys(), key=len, reverse=True)
_SPECIES_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _SPECIES_KEYWORDS_SORTED) + r")\b",
    re.IGNORECASE,
)


def extract_species_from_text(text: str) -> str:
    """Cherche une mention d'espèce dans un texte libre. Retourne 'unknown'
    si aucune espèce reconnue n'est trouvée."""
    match = _SPECIES_PATTERN.search(text)
    if match:
        return SPECIES_MAP[match.group(1).lower()]
    return "unknown"


def normalize_species(value: str) -> str:
    """Normalise une valeur d'espèce déjà connue (colonne structurée)."""
    if value is None:
        return "unknown"
    return SPECIES_MAP.get(value.strip().lower(), "unknown")


# ---------------------------------------------------------------------------
# Normalisation des symptômes (vocabulaire unique entre les 3 sources)
# ---------------------------------------------------------------------------
SYMPTOM_MAP = {
    "appetite loss": "loss of appetite",
    "loss of appetite": "loss of appetite",
    "reduced appetite": "loss of appetite",
    "vomiting": "vomiting",
    "diarrhea": "diarrhea",
    "diarrhoea": "diarrhea",
    "coughing": "coughing",
    "labored breathing": "difficulty breathing",
    "difficulty breathing": "difficulty breathing",
    "lameness": "lameness",
    "skin lesions": "skin lesions",
    "nasal discharge": "nasal discharge",
    "eye discharge": "eye discharge",
    "sneezing": "sneezing",
    "swollen joints": "swollen joints",
    "swollen legs": "swollen joints",
    "swelling": "swelling",
    "weight loss": "weight loss",
    "lethargy": "lethargy",
    "dehydration": "dehydration",
    "decreased milk yield": "decreased milk yield",
    "reduced milk production": "decreased milk yield",
    "reduced wool production": "reduced wool production",
    "reduced wool growth": "reduced wool production",
    "reduced mobility": "reduced mobility",
}


def normalize_symptom(value: str) -> str:
    if value is None:
        return None
    key = value.strip().lower()
    return SYMPTOM_MAP.get(key, key)  # si inconnu, on garde tel quel (en minuscule)


# ---------------------------------------------------------------------------
# Extraction de symptômes depuis du texte libre (Pet Health Symptoms Dataset)
# ---------------------------------------------------------------------------
# Approche heuristique par mots-clés, en attendant un vrai modèle NER entraîné
# (cf. cahier des charges : "Création et entraînement d'un modèle NLP").
# Cette liste est volontairement limitée aux 5 catégories du dataset et devra
# être enrichie/remplacée par un modèle appris dès que possible.
FREE_TEXT_SYMPTOM_KEYWORDS = {
    "vomiting": "vomiting",
    "diarrhea": "diarrhea",
    "diarrhoea": "diarrhea",
    "loss of appetite": "loss of appetite",
    "refuses to eat": "loss of appetite",
    "not eating": "loss of appetite",
    "scratching": "scratching",
    "itching": "itching",
    "flaky": "flaky skin",
    "flaky skin": "flaky skin",
    "hair loss": "hair loss",
    "redness": "redness",
    "swelling": "swelling",
    "swollen": "swelling",
    "discharge": "discharge",
    "limping": "limping",
    "lameness": "lameness",
    "shaking its head": "head shaking",
    "digging at its ears": "ear irritation",
    "lethargy": "lethargy",
    "lethargic": "lethargy",
    "coughing": "coughing",
    "sneezing": "sneezing",
    "weight loss": "weight loss",
    "difficulty walking": "difficulty walking",
    "stiff": "stiffness",
    "stiffness": "stiffness",
    "listless": "lethargy",
    "listlessness": "lethargy",
    "abnormal droppings": "abnormal droppings",
    "crusty": "crusty scales",
    "scales": "crusty scales",
    "struggles to get up": "difficulty rising",
    "struggles to stand": "difficulty rising",
    "difficulty rising": "difficulty rising",
    "recumbent": "difficulty rising",
    "wheezing": "wheezing",
    "clicking": "abnormal breathing sounds",
    "chewing its feathers": "feather plucking",
    "feather plucking": "feather plucking",
    "feather loss": "feather plucking",
    "head shaking": "head shaking",
    "digging at its ears": "ear irritation",
    "crying out in pain": "pain",
    "reluctant to move": "reluctance to move",
    "reluctant to jump": "reluctance to move",
}
_SYMPTOM_KEYWORDS_SORTED = sorted(FREE_TEXT_SYMPTOM_KEYWORDS.keys(), key=len, reverse=True)
_SYMPTOM_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _SYMPTOM_KEYWORDS_SORTED), re.IGNORECASE
)


def extract_symptoms_from_text(text: str) -> list:
    """Recherche des mentions de symptômes connus dans un texte libre.
    Retourne une liste normalisée et dédupliquée (peut être vide)."""
    matches = _SYMPTOM_PATTERN.findall(text)
    normalized = {FREE_TEXT_SYMPTOM_KEYWORDS[m.lower()] for m in matches}
    return sorted(normalized)


# ---------------------------------------------------------------------------
# Normalisation des abréviations médicales (issues de PetEVAL, textes cliniques)
# ---------------------------------------------------------------------------
ABBREVIATION_MAP = {
    "oa": "osteoarthritis",
    "fb": "foreign body",
    "oe": "otitis externa",
    "kc": "kennel cough",
    "uri": "upper respiratory infection",
    "sti": "soft tissue injury",  # à valider avec un vétérinaire : peut aussi
                                  # signifier "sexually transmitted infection"
                                  # selon le contexte animal — ambigu, à vérifier
    "cystitis": "cystitis",
    "mvd": "mitral valve disease",
    "spey": "spay",
    "biop": "biopsy",
    "tx": "treatment",
    "vax": "vaccination",
}


def normalize_disease_term(entity_text: str) -> str:
    key = entity_text.strip().lower()
    return ABBREVIATION_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Groupes d'âge (dérivés du champ 'age' en années)
# ---------------------------------------------------------------------------
def derive_age_group(age, species: str = "unknown") -> str:
    if age is None:
        return "unknown"
    if age <= 1:
        return "young"      # chiot/chaton/poulain selon l'espèce
    if age >= 8:
        return "senior"
    return "adult"