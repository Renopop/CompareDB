# Guide Utilisateur CompareDB

## Table des matières

1. [Configuration de base](#configuration-de-base)
2. [Modes d'exécution](#modes-dexécution)
3. [Stratégie combinatoire](#stratégie-combinatoire)
4. [Analyse LLM](#analyse-llm)
5. [Paramètres avancés](#paramètres-avancés)
6. [Interprétation des résultats](#interprétation-des-résultats)

---

## Configuration de base

### Sélection des fichiers

Deux modes disponibles :

- **Upload/Parcourir** : Glisser-déposer ou cliquer pour sélectionner
- **Saisie manuelle** : Entrer le chemin complet (ex: `C:\Data\fichier.xlsx`)

Pour chaque fichier, spécifier :
- **Nom de la feuille** : Nom exact (ex: "Feuil1")
- **Numéro de colonne** : 1, 2, 3...

---

## Modes d'exécution

### Mode en ligne (par défaut)

- APIs distantes (Snowflake + DALLEM)
- Rapide et puissant
- Nécessite connexion réseau

### Mode hors ligne

- Modèles locaux installés sur la machine
- Fonctionne sans internet
- Plus lent mais autonome

**Prérequis hors ligne :**
```bash
pip install torch transformers sentence-transformers accelerate
```

**Configuration** dans `offline_models.py` :
```python
AVAILABLE_LLM_MODELS = {
    "qwen": "C:\\Models\\Qwen\\Qwen2.5-3B-Instruct",
    "mistral": "C:\\Models\\mistralai\\Mistral-7B-Instruct-v0.3",
}
DEFAULT_EMBEDDING = "C:\\Models\\BAAI\\bge-m3"
```

---

## Stratégie combinatoire

### Principe

Pour chaque mismatch, tente de combiner plusieurs lignes de Base 2 pour trouver une correspondance.

### Schéma détaillé

```
Mismatch de Base 1 : "Le système doit supporter 1000 utilisateurs avec temps < 2s"
                                    │
                    ┌───────────────┴───────────────┐
                    │  Comparer avec TOUTE Base 2   │
                    └───────────────┬───────────────┘
                                    │
            Scores : [0.65, 0.52, 0.71, 0.48, 0.60, ...]
                                    │
                    ┌───────────────┴───────────────┐
                    │  Trier par score décroissant  │
                    └───────────────┬───────────────┘
                                    │
            Top-k : [0.71, 0.65, 0.60, 0.52, ...]
                                    │
┌───────────────────────────────────┴───────────────────────────────────┐
│                    COMBINAISONS TESTÉES                               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  k=2 : Combiner lignes [idx_3, idx_1]                                │
│       "Le démarrage est rapide" + "Temps de réponse optimal"         │
│       → Similarité = 0.73 < 0.78 (seuil) ✗                           │
│                                                                       │
│  k=3 : Combiner lignes [idx_3, idx_1, idx_5]                         │
│       "Le démarrage est rapide" +                                    │
│       "Temps de réponse optimal" +                                   │
│       "Performance en moins de 5s"                                   │
│       → Similarité = 0.82 >= 0.78 (seuil) ✓ MATCH TROUVÉ!           │
│                                                                       │
│  → Arrêt, match combinatoire créé                                    │
│                                                                       │
│  Résultat :                                                          │
│  - tgt_indices_combined: [3, 1, 5]                                   │
│  - score: 0.82                                                       │
│  - match_type: "combinatorial"                                       │
│  - warning: "MATCH COMBINATOIRE : Lignes combinées = [3, 1, 5]"      │
└───────────────────────────────────────────────────────────────────────┘
```

### Activation

Cocher **"Stratégie combinatoire pour mismatches"** dans les paramètres avancés.

**Paramètre** : Nombre max de combinaisons (2-5, défaut: 4)

---

## Analyse LLM

### Fonctionnement

Pour chaque match, le LLM examine les textes et répond :
- **TRUE** : Textes sémantiquement équivalents
- **FALSE** : Textes non équivalents
- **Commentaire** : Explication de la décision

### Exemple

```
Source : "Le moteur doit démarrer en moins de 5 secondes"
Target : "Temps de démarrage < 5s"

→ équivalence: TRUE
→ commentaire: "Les deux textes expriment la même exigence de performance
               au démarrage, avec la même contrainte de 5 secondes"
```

### Activation

- Cocher **"Analyse LLM des équivalences"**
- Auto-activé si stratégie combinatoire active

### Budget LLM

Limite le nombre d'analyses (défaut: 200, max: 1000).

Les matches sont analysés par ordre de score décroissant.

---

## Paramètres avancés

| Paramètre | Description | Défaut | Recommandation |
|-----------|-------------|--------|----------------|
| **Seuil de similarité** | Score minimum pour match | 0.78 | 0.75-0.82 selon besoin |
| **Taille de batch** | Éléments traités simultanément | 16 | 8-16 CPU, 32-64 GPU |
| **Limite de lignes** | Max lignes à traiter (0=tout) | 0 | Tester avec 100 d'abord |
| **Mode de matching** | Complet ou Approximatif | Complet | Approximatif si >10k lignes |

---

## Interprétation des résultats

### Métriques affichées

**Sans combinatoire :**
```
┌───────────────────┬───────────────────┬──────────────────┐
│ ✓ Matches         │ ✗ Sous le seuil   │ Taux de match    │
│      150          │        45         │      76.9%       │
└───────────────────┴───────────────────┴──────────────────┘
```

**Avec combinatoire :**
```
┌──────────────┬─────────────────┬───────────────────┬──────────────────┐
│ ✓ Matches    │ ✓ Matches       │ ✗ Mismatches      │ Taux de match    │
│   normaux    │  combinatoires  │   définitifs      │                  │
│     150      │       23        │        22         │      88.7%       │
└──────────────┴─────────────────┴───────────────────┴──────────────────┘
```

### Colonnes des fichiers Excel

#### matches_*.xlsx

| Colonne | Description |
|---------|-------------|
| `src_index` | Index ligne Base 1 |
| `tgt_index` | Index ligne Base 2 (null si combinatoire) |
| `source` | Texte Base 1 |
| `target` | Texte Base 2 |
| `score` | Score de similarité |
| `match_type` | "normal" ou "combinatorial" |
| `équivalence` | TRUE/FALSE/None |
| `commentaire` | Explication LLM |
| `warning` | Avertissement (si combinatoire) |
| `tgt_indices_combined` | Indices combinés |

#### under_*.xlsx

Mismatches définitifs (aucune correspondance trouvée).

### Signification des colonnes LLM

| Colonne `analyse_llm` | Signification |
|-----------------------|---------------|
| "Oui (normal)" | Match normal analysé |
| "Oui (combinatoire)" | Match combinatoire analysé |
| "Non" | Non analysé (limite budget) |

| Colonne `équivalence` | Signification |
|-----------------------|---------------|
| TRUE | LLM confirme l'équivalence |
| FALSE | LLM rejette l'équivalence |
| None | Non analysé |

---

## Résolution de problèmes

| Problème | Solution |
|----------|----------|
| Trop de faux positifs | Augmenter seuil (0.82-0.85), activer LLM |
| Trop de faux négatifs | Diminuer seuil (0.72-0.75), activer combinatoire |
| Traitement très lent | Mode approximatif, réduire budget LLM |
| Matches combinatoires non pertinents | Réduire max_combinations |

---

**Version :** 2.0 | **Mise à jour :** 2025-11-25
