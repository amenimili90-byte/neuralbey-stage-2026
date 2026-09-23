# -*- coding: utf-8 -*-
"""
Phase 5 - Localisation des vétérinaires (Tunisie)

Cahier des charges, section 4.1 : "Gestion des cas urgents (redirection
immédiate vers un vétérinaire)". Doctobot étant utilisé uniquement en
Tunisie, cette redirection doit pointer vers un vétérinaire réel et
proche de l'utilisateur, pas juste un message générique.

Source des données : Google Places (recherche manuelle par gouvernorat),
compilée dans vets_tunisia.csv. Couverture : au moins une clinique par
gouvernorat (24/24), avec nom, adresse, téléphone (quand disponible),
coordonnées GPS et note.

NOTE IMPORTANTE : cette base est un point de départ (1 à 3 cliniques par
gouvernorat), pas un annuaire exhaustif. Pour une mise en production
réelle, il faudrait soit l'enrichir significativement (plus de cliniques
par région), soit la brancher sur une API cartographique en direct.
"""

import csv
import re
import unicodedata
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

CSV_PATH = Path(__file__).parent / "vets_tunisia.csv"

# Alias courants -> nom exact du gouvernorat tel qu'écrit dans le CSV
GOVERNORATE_ALIASES = {
    "kef": "Le Kef",
    "el kef": "Le Kef",
    "medenine": "Médenine",
    "gabes": "Gabès",
    "beja": "Béja",
    "sfax": "Sfax",
    "tunis": "Tunis",
    "ariana": "Ariana",
    "manouba": "Manouba",
    "ben arous": "Ben Arous",
    "nabeul": "Nabeul",
    "zaghouan": "Zaghouan",
    "bizerte": "Bizerte",
    "jendouba": "Jendouba",
    "siliana": "Siliana",
    "kasserine": "Kasserine",
    "sidi bouzid": "Sidi Bouzid",
    "sousse": "Sousse",
    "monastir": "Monastir",
    "mahdia": "Mahdia",
    "kairouan": "Kairouan",
    "gafsa": "Gafsa",
    "tozeur": "Tozeur",
    "kebili": "Kebili",
    "tataouine": "Tataouine",
}


def normalize(text):
    """Supprime les accents et met en minuscule, pour comparer sans se
    soucier de "Béja" vs "beja" vs "Beja"."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def load_vets():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


VETS = load_vets()
NORMALIZED_ALIASES = {normalize(k): v for k, v in GOVERNORATE_ALIASES.items()}


def resolve_governorate(user_text):
    """Essaie de retrouver un nom de gouvernorat valide à partir d'un
    texte libre donné par l'utilisateur (ville, quartier, gouvernorat)."""
    norm = normalize(user_text)

    # 1. Correspondance directe sur un alias de gouvernorat
    for alias_norm, governorate in NORMALIZED_ALIASES.items():
        if alias_norm in norm:
            return governorate

    # 2. Correspondance sur le nom de ville/quartier dans la base
    for vet in VETS:
        if normalize(vet["city"]) in norm or norm in normalize(vet["city"]):
            return vet["governorate"]

    return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance approximative en kilomètres entre deux points GPS."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_vets(user_location_text, user_lat=None, user_lon=None, top_k=3):
    """Retourne jusqu'à top_k cliniques pertinentes selon la localisation
    donnée par l'utilisateur (texte libre : ville, gouvernorat...).

    Si user_lat/user_lon sont fournis (ex. géolocalisation d'une future
    version app mobile de Cheebo), les résultats sont triés par distance
    réelle. Sinon, on filtre par gouvernorat et on trie par note."""
    governorate = resolve_governorate(user_location_text)

    if governorate:
        candidates = [v for v in VETS if v["governorate"] == governorate]
    else:
        candidates = VETS  # fallback : pas de gouvernorat identifié

    if user_lat is not None and user_lon is not None:
        candidates = [c for c in candidates if c["latitude"] and c["longitude"]]
        candidates.sort(
            key=lambda c: haversine_km(user_lat, user_lon, float(c["latitude"]), float(c["longitude"]))
        )
    else:
        candidates.sort(key=lambda c: float(c["rating"]) if c["rating"] else 0, reverse=True)

    return candidates[:top_k], governorate


def format_vet_line(vet):
    phone = vet["phone"] if vet["phone"] else "téléphone non disponible"
    return f"{vet['name']} — {vet['address']} — {phone}"


if __name__ == "__main__":
    # Petit test manuel
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Sousse"
    results, gov = find_vets(query)
    print(f"Gouvernorat détecté : {gov}")
    for v in results:
        print(" -", format_vet_line(v))
