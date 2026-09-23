# Schéma JSON commun — Doctobot

Chaque enregistrement, quelle que soit sa source d'origine, sera converti
vers ce format unique.

```json
{
  "id": "ph_00001",
  "source": "pet_health_symptoms",
  "text": "My rabbit has white, flaky material inside the ear canals.",
  "species": "rabbit",
  "age": null,
  "age_group": "unknown",
  "symptoms": ["ear discharge", "flaky skin"],
  "duration": null,
  "disease_label": null,
  "condition_category": "Ear Infections",
  "icd_category": [],
  "severity": "unknown",
  "record_type": "owner_observation"
}
```

## Description des champs

| Champ | Type | Description | Rempli par |
|---|---|---|---|
| `id` | string | identifiant unique préfixé par la source (`ph_`, `pe_`, `ad_`) | généré |
| `source` | string | `pet_health_symptoms` / `peteval` / `animal_disease_prediction` | fixe |
| `text` | string | texte brut original (pour le RAG) | copié tel quel |
| `species` | string | espèce normalisée : `dog`, `cat`, `rabbit`, `cow`, `goat`, `horse`, `pig`, `sheep`, `unknown` | direct (Animal Disease) ou extrait du texte (les 2 autres) |
| `age` | int / null | âge en années si connu | direct (Animal Disease) ou `null` |
| `age_group` | string | `puppy/kitten`, `adult`, `senior`, `unknown` — dérivé de `age` | calculé |
| `symptoms` | liste de strings | symptômes normalisés (vocabulaire unique) | direct ou extrait/NER |
| `duration` | string / null | durée des symptômes si connue (ex. `"2 days"`) | direct (Animal Disease) ou `null` |
| `disease_label` | string / null | maladie précise si identifiée (ex. `"Parvovirus"`, `"otitis"`) | direct ou NER |
| `condition_category` | string / null | catégorie large (5 catégories Pet Health) | direct si disponible |
| `icd_category` | liste de strings | catégories ICD-11 (PetEVAL uniquement) | direct |
| `severity` | string | `low`, `medium`, `high`, `unknown` — dérivé via la base de règles de triage | calculé |
| `record_type` | string | `owner_observation`, `clinical_notes`, `clinical_free_text`, `structured_record` | direct ou fixe |

## Pourquoi ce schéma et pas un autre

- **`text` toujours conservé** : même pour le dataset structuré, on reconstruit une phrase à partir des colonnes (utile pour le RAG et pour l'entraînement d'un futur modèle NLU unifié).
- **`symptoms` toujours en liste normalisée**, jamais en 4 colonnes séparées (`Symptom_1-4`) — évite la redondance identifiée dans Animal Disease Prediction.
- **`icd_category` séparé de `condition_category`** : ce sont deux taxonomies différentes (ICD-11 vs les 5 catégories custom de Pet Health), on ne les fusionne pas artificiellement.
- **`severity` calculé, pas copié** : aucune des 3 sources ne donne directement un niveau de gravité — il sera dérivé via la base de règles de triage (symptômes critiques, durée) qu'on avait esquissée au tout début.
