"""
Doctobot - Base de règles de triage (severity)
==================================================

 AVERTISSEMENT IMPORTANT
Cette base de règles est une PROPOSITION TECHNIQUE DE DÉPART, construite à
partir du vocabulaire observé dans les 3 datasets. Elle n'a AUCUNE validation
médicale. Avant toute utilisation réelle dans Doctobot, elle doit être
relue et corrigée par un vétérinaire qualifié. Ne pas déployer tel quel.

Niveaux : "high" (urgence -> consulter immédiatement), "medium" (consulter
sous peu), "low" (surveillance), "unknown" (signal insuffisant pour juger).
"""

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
UNKNOWN = "unknown"

_SEVERITY_ORDER = {UNKNOWN: 0, LOW: 1, MEDIUM: 2, HIGH: 3}

# --- Gravité associée à un symptôme normalisé -------------------------------
SYMPTOM_SEVERITY = {
    "difficulty breathing": HIGH,
    "wheezing": HIGH,
    "abnormal breathing sounds": HIGH,
    "dehydration": HIGH,
    "foreign body": HIGH,
    "pain": MEDIUM,
    "vomiting": MEDIUM,
    "diarrhea": MEDIUM,
    "lethargy": MEDIUM,
    "fever": MEDIUM,
    "loss of appetite": MEDIUM,
    "weight loss": MEDIUM,
    "swelling": MEDIUM,
    "abnormal droppings": MEDIUM,
    "difficulty rising": MEDIUM,
    "reluctance to move": MEDIUM,
    "lameness": MEDIUM,
    "difficulty walking": MEDIUM,
    "reduced mobility": MEDIUM,
    "swollen joints": MEDIUM,
    "ulcer": MEDIUM,
    "abscess": MEDIUM,
    "cystitis": MEDIUM,
    "upper respiratory infection": MEDIUM,
    "mitral valve disease": HIGH,
    "soft tissue injury": MEDIUM,
    "coughing": LOW,
    "sneezing": LOW,
    "itching": LOW,
    "scratching": LOW,
    "hair loss": LOW,
    "crusty scales": LOW,
    "flaky skin": LOW,
    "redness": LOW,
    "discharge": LOW,
    "ear irritation": LOW,
    "head shaking": LOW,
    "feather plucking": LOW,
    "stiffness": LOW,
    "gingivitis": LOW,
    "dental disease": LOW,
    "conjunctivitis": LOW,
    "allergy": LOW,
    "lipoma": LOW,
    "osteoarthritis": LOW,
    "otitis externa": LOW,
    "otitis": LOW,
    "kennel cough": LOW,
    "infection": MEDIUM,  # générique, prudence
}

# --- Gravité associée à une maladie précise (colonne disease_label) --------
DISEASE_SEVERITY = {
    "parvovirus": HIGH,
    "foot and mouth disease": HIGH,
    "swine fever": HIGH,
    "gastroenteritis": MEDIUM,
}

# Durées considérées comme "prolongées" -> on augmente la gravité d'un cran
_LONG_DURATION_KEYWORDS = ["week", "weeks"]


def _escalate(level: str) -> str:
    order = [UNKNOWN, LOW, MEDIUM, HIGH]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]


def compute_severity(record: dict) -> str:
    """Calcule le niveau de gravité d'un enregistrement à partir de ses
    symptômes et, si connu, de sa maladie précise. Règle : on retient le
    niveau le PLUS ÉLEVÉ trouvé parmi tous les signaux disponibles.

    L'escalade par durée prolongée est volontairement limitée à low->medium :
    un symptôme bénin qui persiste mérite une vraie consultation, mais la
    durée seule ne doit jamais transformer un cas "medium" en urgence
    ("high") — seuls des symptômes intrinsèquement graves le justifient.
    Sans cette limite, un cas comme "léthargie + lésions cutanées depuis
    2 semaines" (medium) serait à tort classé "high", au même niveau qu'une
    détresse respiratoire."""
    best = UNKNOWN

    disease = (record.get("disease_label") or "").strip().lower()
    if disease in DISEASE_SEVERITY:
        candidate = DISEASE_SEVERITY[disease]
        if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[best]:
            best = candidate

    for symptom in record.get("symptoms", []):
        candidate = SYMPTOM_SEVERITY.get(symptom.lower())
        if candidate and _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[best]:
            best = candidate

    duration = (record.get("duration") or "").lower()
    if best == LOW and any(kw in duration for kw in _LONG_DURATION_KEYWORDS):
        best = MEDIUM  # low -> medium uniquement, jamais medium -> high

    return best