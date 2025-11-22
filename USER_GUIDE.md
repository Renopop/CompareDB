# 📘 Guide Utilisateur CompareDB

## 📋 Table des matières

1. [Introduction](#introduction)
2. [Schéma de la méthode de comparaison](#schéma-de-la-méthode-de-comparaison)
3. [Configuration de base](#configuration-de-base)
4. [Modes d'exécution](#modes-dexécution)
5. [Stratégies de matching](#stratégies-de-matching)
6. [Analyse LLM](#analyse-llm)
7. [Interprétation des résultats](#interprétation-des-résultats)
8. [Paramètres avancés](#paramètres-avancés)
9. [Exemples pratiques](#exemples-pratiques)

---

## 🎯 Introduction

**CompareDB** est un outil de comparaison sémantique intelligent qui permet de comparer deux colonnes Excel en utilisant l'intelligence artificielle pour détecter des équivalences au-delà de la simple correspondance textuelle.

### Cas d'usage typiques

- Comparaison de spécifications techniques entre versions
- Détection d'équivalences sémantiques dans des bases de données
- Analyse de cohérence entre documents
- Matching de requirements entre systèmes

---

## 🔄 Schéma de la méthode de comparaison

### Vue d'ensemble du processus

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ÉTAPE 1 : CHARGEMENT DES DONNÉES                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              ┌─────▼─────┐                   ┌─────▼─────┐
              │  Base 1   │                   │  Base 2   │
              │ (Source)  │                   │ (Target)  │
              └─────┬─────┘                   └─────┬─────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│              ÉTAPE 2 : GÉNÉRATION DES EMBEDDINGS (IA)                   │
│                                                                          │
│  Chaque ligne est transformée en vecteur numérique (embedding)          │
│  qui capture son sens sémantique                                        │
│                                                                          │
│  "Le moteur démarre" → [0.23, -0.45, 0.78, ... ] (1024 dimensions)    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│           ÉTAPE 3 : MATCHING PAR SIMILARITÉ COSINUS                     │
│                                                                          │
│  Pour chaque ligne de Base 1, on calcule la similarité avec            │
│  toutes les lignes de Base 2                                           │
│                                                                          │
│  Similarité = cos(θ) entre les vecteurs                                │
│  Score ∈ [0, 1] : 1 = identique, 0 = totalement différent             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
          Score ≥ Seuil ?                  Score < Seuil ?
                    │                               │
              ┌─────▼─────┐                   ┌─────▼─────┐
              │  MATCHES  │                   │ MISMATCHES│
              │  NORMAUX  │                   │           │
              └─────┬─────┘                   └─────┬─────┘
                    │                               │
                    │                               │
┌───────────────────┴──────────┐                    │
│  ÉTAPE 4a : ANALYSE LLM      │                    │
│  (optionnel)                 │                    │
│                              │                    │
│  Le LLM examine chaque match │                    │
│  et détermine si c'est une   │                    │
│  vraie équivalence           │                    │
│                              │                    │
│  Résultat :                  │                    │
│  - ✅ TRUE (équivalent)      │                    │
│  - ❌ FALSE (non équivalent) │                    │
│  - ⚠️ Incertain              │                    │
└───────────────────┬──────────┘                    │
                    │                               │
                    │              ┌────────────────┴────────────────┐
                    │              │ ÉTAPE 4b : STRATÉGIE            │
                    │              │ COMBINATOIRE (optionnel)        │
                    │              │                                 │
                    │              │ Pour chaque mismatch :          │
                    │              │                                 │
                    │              │ 1. Comparer avec TOUTE Base 2   │
                    │              │ 2. Prendre top-k lignes         │
                    │              │ 3. Combiner les textes          │
                    │              │ 4. Recalculer similarité        │
                    │              │ 5. Si ≥ seuil → Match combinatoire │
                    │              │ 6. Sinon, essayer k+1 lignes    │
                    │              │ 7. Jusqu'à max_combinations     │
                    │              │                                 │
                    │              └────────┬────────────────────────┘
                    │                       │
                    │         ┌─────────────┴─────────────┐
                    │         │                           │
                    │    ┌────▼────┐              ┌───────▼──────┐
                    │    │ MATCHES │              │  MISMATCHES  │
                    │    │COMBINA- │              │  DÉFINITIFS  │
                    │    │ TOIRES  │              │              │
                    │    └────┬────┘              └──────────────┘
                    │         │
                    │         │
                    │    ┌────▼─────────────────────┐
                    │    │ ÉTAPE 4c : ANALYSE LLM   │
                    │    │ DES MATCHES COMBINATOIRES│
                    │    │                          │
                    │    │ Validation automatique   │
                    │    │ de chaque match trouvé   │
                    │    └────┬─────────────────────┘
                    │         │
                    └─────────┴──────────┐
                                         │
┌────────────────────────────────────────▼─────────────────────────────────┐
│                    ÉTAPE 5 : EXPORT DES RÉSULTATS                        │
│                                                                           │
│  📊 Fichiers Excel générés :                                            │
│  - matches_YYYYMMDD_HHMMSS.xlsx (tous les matches)                      │
│  - under_YYYYMMDD_HHMMSS.xlsx (mismatches définitifs)                   │
│                                                                           │
│  📋 Colonnes incluses :                                                  │
│  - src_index, tgt_index : Indices des lignes                            │
│  - source, target : Textes comparés                                     │
│  - score : Score de similarité                                          │
│  - match_type : normal / combinatorial / definitive_mismatch            │
│  - équivalence : TRUE/FALSE/None (si analyse LLM)                       │
│  - commentaire : Explication du LLM                                     │
│  - analyse_llm : "Oui (normal)" / "Oui (combinatoire)" / "Non"         │
│  - warning : Message d'alerte (pour matches combinatoires)             │
│  - tgt_indices_combined : Liste des indices combinés (si combinatoire) │
└──────────────────────────────────────────────────────────────────────────┘
```

### Détail de la stratégie combinatoire

```
Mismatch de Base 1 : "Le système doit démarrer en moins de 5 secondes"
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
│  k=2 : Combiner lignes [idx_3, idx_1]                               │
│       "Le démarrage est rapide" + "Temps de réponse optimal"        │
│       → Similarité = 0.73 < 0.78 (seuil) ❌                         │
│                                                                       │
│  k=3 : Combiner lignes [idx_3, idx_1, idx_5]                        │
│       "Le démarrage est rapide" +                                    │
│       "Temps de réponse optimal" +                                   │
│       "Performance en moins de 5s"                                   │
│       → Similarité = 0.82 ≥ 0.78 (seuil) ✅ MATCH TROUVÉ!          │
│                                                                       │
│  → Arrêt de la recherche, match combinatoire créé                   │
│                                                                       │
│  Résultat :                                                          │
│  - src_index: 42                                                     │
│  - tgt_indices_combined: [3, 1, 5]                                  │
│  - target: "Le démarrage est rapide Temps de réponse..."            │
│  - score: 0.82                                                       │
│  - match_type: "combinatorial"                                       │
│  - warning: "⚠️ MATCH COMBINATOIRE : Lignes base 2 combinées = [3, 1, 5]" │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   ANALYSE LLM AUTOMATIQUE     │
                    │                               │
                    │  Question au LLM :            │
                    │  "Est-ce que ces textes       │
                    │   sont équivalents ?"         │
                    │                               │
                    │  Réponse :                    │
                    │  - équivalence: TRUE          │
                    │  - commentaire: "Les deux     │
                    │    exigences concernent le    │
                    │    temps de démarrage..."     │
                    └───────────────────────────────┘
```

---

## ⚙️ Configuration de base

### 1. Sélection des fichiers

Deux modes disponibles :

#### 📤 Mode Upload/Parcourir
- Glisser-déposer le fichier Excel directement
- Ou cliquer pour parcourir vos fichiers
- Fichiers supportés : `.xlsx`, `.xls`

#### ⌨️ Mode Saisie manuelle
- Entrer le chemin complet du fichier
- Exemple : `L:\Test\Classeur1.xlsx`

### 2. Configuration des colonnes

Pour chaque fichier :
- **Nom de la feuille** : Nom exact de la feuille Excel (ex: "Feuil1")
- **Numéro de colonne** : Numéro de la colonne à comparer (1, 2, 3...)

---

## 🌐 Modes d'exécution

### Mode en ligne (par défaut)

**Caractéristiques :**
- Utilise les API distantes (Snowflake + DALLEM)
- Modèles puissants et rapides
- Nécessite une connexion réseau
- Recommandé pour la production

**Modèles utilisés :**
- Embedding : `snowflake-arctic-embed-l-v2.0`
- LLM : `dallem-val`

### Mode hors ligne

**Caractéristiques :**
- Utilise des modèles locaux installés sur votre machine
- Fonctionne sans connexion internet
- Plus lent mais autonome
- Nécessite l'installation de dépendances supplémentaires

**Modèles disponibles :**
- **LLM :**
  - Qwen 2.5 3B Instruct (léger, rapide)
  - Mistral 7B Instruct v0.3 (plus performant)
- **Embedding :**
  - BGE-M3 (multilingue)

**Prérequis :**
```bash
pip install torch transformers sentence-transformers accelerate
```

**Configuration :**
Modifier les chemins dans `offline_models.py` :
```python
AVAILABLE_LLM_MODELS = {
    "qwen": "D:\\IA Test\\models\\Qwen\\Qwen2.5-3B-Instruct",
    "mistral": "D:\\IA Test\\models\\mistralai\\Mistral-7B-Instruct-v0.3",
}
```

---

## 🎯 Stratégies de matching

### 1. Matching normal (toujours actif)

**Principe :**
- Chaque ligne de Base 1 est comparée avec toutes les lignes de Base 2
- Le meilleur score est retenu
- Si score ≥ seuil → Match
- Si score < seuil → Mismatch

**Exemple :**
```
Base 1 : "Le moteur doit démarrer"
Base 2 : "Démarrage du moteur requis"

Similarité = 0.85 ≥ 0.78 (seuil) → MATCH ✅
```

### 2. Stratégie combinatoire (optionnelle)

**Activation :**
Cocher ✅ `🔀 Stratégie combinatoire pour mismatches`

**Principe :**
Pour chaque mismatch, tente de combiner plusieurs lignes de Base 2 pour trouver une correspondance.

**Fonctionnement détaillé :**

1. **Prendre un mismatch** de Base 1
   ```
   Ligne 42 : "Le système doit supporter 1000 utilisateurs simultanés avec temps de réponse < 2s"
   ```

2. **Calculer similarité avec TOUTE Base 2**
   ```
   Base 2[15] : "Support de 1000 users" → Score: 0.68
   Base 2[23] : "Temps de réponse rapide" → Score: 0.62
   Base 2[31] : "Performance < 2 secondes" → Score: 0.65
   Base 2[8]  : "Capacité utilisateurs" → Score: 0.58
   ```

3. **Trier par score décroissant**
   ```
   Top-4 : [15, 31, 23, 8]
   ```

4. **Tester combinaisons k=2, 3, 4...**

   **k=2 :**
   ```
   Combiner [15, 31] :
   "Support de 1000 users Performance < 2 secondes"
   → Similarité = 0.76 < 0.78 ❌
   ```

   **k=3 :**
   ```
   Combiner [15, 31, 23] :
   "Support de 1000 users Performance < 2 secondes Temps de réponse rapide"
   → Similarité = 0.81 ≥ 0.78 ✅ MATCH TROUVÉ!
   ```

5. **Créer le match combinatoire**
   ```
   Match combinatoire créé :
   - Source : "Le système doit supporter 1000 utilisateurs..."
   - Target : "Support de 1000 users Performance < 2 secondes Temps de réponse rapide"
   - Score : 0.81
   - Warning : ⚠️ MATCH COMBINATOIRE : Lignes base 2 combinées = [15, 31, 23]
   ```

**Paramètres :**
- **Nombre max de combinaisons** : Limite à combien de lignes peuvent être combinées
  - Min: 2 (combine 2 lignes maximum)
  - Max: 5 (combine jusqu'à 5 lignes)
  - Défaut: 4

**Recommandations :**
- Utiliser pour des bases où une exigence peut être éclatée en plusieurs lignes
- Augmenter max_combinations pour des bases très fragmentées
- Les matches combinatoires sont automatiquement analysés par LLM

---

## 🤖 Analyse LLM

### Activation

**Analyse LLM des équivalences :**
Cocher ✅ `🔍 Analyse LLM des équivalences`

**Activation automatique :**
Le LLM est **automatiquement activé** si vous activez la stratégie combinatoire (nécessaire pour valider les combinaisons).

### Fonctionnement

Pour chaque match (normal ou sous le seuil), le LLM examine :
1. Le texte source (Base 1)
2. Le texte cible (Base 2)

Et répond :
- **TRUE** : Les textes sont équivalents sémantiquement
- **FALSE** : Les textes ne sont PAS équivalents
- **Commentaire** : Explication de la décision

### Exemple d'analyse

```
Source : "Le moteur doit démarrer en moins de 5 secondes"
Target : "Temps de démarrage < 5s"

→ équivalence: TRUE
→ commentaire: "Les deux textes expriment la même exigence de performance
               au démarrage, avec la même contrainte temporelle de 5 secondes"
```

### Promotion automatique

Si un mismatch (score < seuil) est validé par le LLM comme équivalent :
- Il est **promu** en match
- La colonne `promu_par_llm` = TRUE
- Il apparaît dans le fichier `matches_*.xlsx`

### Budget LLM

**Nombre max d'analyses LLM :**
- Par défaut : 200
- Plage : 1 à 1000

**Ordre d'analyse :**
1. Matches normaux (score ≥ seuil) - par score décroissant
2. Mismatches (score < seuil) - par score décroissant
3. Matches combinatoires (tous analysés automatiquement)

**Raison du budget :**
- Les appels LLM sont coûteux en temps/ressources
- Permet de limiter le temps de traitement
- Les matches les plus probables sont analysés en premier

---

## 📊 Interprétation des résultats

### Métriques affichées

#### Sans stratégie combinatoire :
```
┌───────────────────┬───────────────────┬──────────────────┐
│ ✅ Matches        │ ⚠️ Sous le seuil │ 📊 Taux de match │
│      150          │        45         │      76.9%       │
└───────────────────┴───────────────────┴──────────────────┘
```

#### Avec stratégie combinatoire :
```
┌──────────────┬─────────────────┬───────────────────┬──────────────────┐
│ ✅ Matches   │ 🔀 Matches      │ ⚠️ Mismatches     │ 📊 Taux de match │
│   normaux    │  combinatoires  │   définitifs      │                  │
│     150      │       23        │        22         │      88.7%       │
└──────────────┴─────────────────┴───────────────────┴──────────────────┘
```

### Fichiers Excel générés

#### 1. `matches_YYYYMMDD_HHMMSS.xlsx`

**Contenu :** Tous les matches (normaux + combinatoires)

**Colonnes principales :**
| Colonne | Description | Exemple |
|---------|-------------|---------|
| `src_index` | Index ligne Base 1 | 42 |
| `tgt_index` | Index ligne Base 2 (null si combinatoire) | 15 ou null |
| `source` | Texte Base 1 | "Le moteur doit démarrer" |
| `target` | Texte Base 2 (combiné si combinatoire) | "Démarrage du moteur requis" |
| `score` | Score de similarité | 0.85 |
| `match_type` | Type de match | "normal" ou "combinatorial" |
| `équivalence` | Validation LLM | TRUE / FALSE / None |
| `commentaire` | Explication LLM | "Les deux textes..." |
| `analyse_llm` | Analyse effectuée ? | "Oui (normal)" / "Non" |
| `promu_par_llm` | Promu depuis mismatch ? | TRUE / FALSE |
| `warning` | Avertissement (si combinatoire) | "⚠️ MATCH COMBINATOIRE..." |
| `tgt_indices_combined` | Indices combinés | [15, 31, 23] |
| `combination_size` | Nombre de lignes combinées | 3 |
| `individual_scores` | Scores individuels | [0.68, 0.65, 0.62] |

#### 2. `under_YYYYMMDD_HHMMSS.xlsx`

**Contenu :** Mismatches définitifs (aucune correspondance trouvée)

**Colonnes principales :**
| Colonne | Description |
|---------|-------------|
| `src_index` | Index ligne Base 1 |
| `source` | Texte Base 1 |
| `target` | Meilleur match trouvé (même s'il est mauvais) |
| `score` | Meilleur score (< seuil) |
| `match_type` | "definitive_mismatch" |
| `équivalence` | FALSE (si analysé par LLM) |
| `commentaire` | Explication LLM pourquoi pas équivalent |

### Interprétation des résultats LLM

#### Colonne `analyse_llm`
- **"Oui (normal)"** : Match normal analysé par LLM
- **"Oui (combinatoire)"** : Match combinatoire analysé par LLM
- **"Oui (mismatch)"** : Mismatch analysé par LLM
- **"Non"** : Non analysé (limite budget atteinte)

#### Colonne `équivalence`
- **TRUE** ✅ : LLM confirme l'équivalence
- **FALSE** ❌ : LLM rejette l'équivalence
- **None** ⚠️ : Non analysé ou réponse incertaine

#### Validation des matches combinatoires

Après la stratégie combinatoire, l'interface affiche :
```
📊 Résultats LLM : 18 validés ✅, 3 rejetés ❌, 2 incertains ⚠️
```

- **Validés** : Le LLM confirme que la combinaison est pertinente
- **Rejetés** : La combinaison n'a pas de sens sémantique
- **Incertains** : Le LLM n'a pas tranché (équivalence = None)

---

## 🔧 Paramètres avancés

### Seuil de similarité

**Définition :** Score minimum pour considérer un match

**Plage :** 0.0 à 1.0
- **0.0** : Tout matche (trop permissif)
- **1.0** : Seuls les textes identiques matchent (trop strict)
- **0.78** (défaut) : Bon équilibre

**Recommandations :**
- **0.70 - 0.75** : Matching souple, plus de faux positifs
- **0.78 - 0.82** : Équilibre précision/recall
- **0.85 - 0.90** : Matching strict, plus de faux négatifs

### Taille de batch

**Définition :** Nombre d'éléments traités simultanément lors de la génération des embeddings

**Plage :** 8 à 128
- **Défaut : 16**

**Impact :**
- **Petit (8-16)** : Moins de mémoire, plus lent
- **Grand (64-128)** : Plus de mémoire, plus rapide

**Recommandations :**
- **CPU** : 8-16
- **GPU** : 32-64
- **GPU puissant** : 64-128

### Limite de lignes

**Définition :** Nombre maximum de lignes à traiter (pour tests)

**Usage :**
- **0** : Traiter toutes les lignes (défaut)
- **100** : Tester avec 100 premières lignes
- Utile pour valider la configuration avant traitement complet

### Mode de matching

#### Complet (défaut)
- Matrice complète de similarité
- Chaque ligne de Base 1 comparée avec TOUTES les lignes de Base 2
- Plus précis
- Plus lent pour grandes bases

#### Approximatif (top-k)
- Recherche approximative des k meilleurs candidats
- Plus rapide
- Légère perte de précision

**Recommandations :**
- **< 10,000 lignes** : Mode complet
- **> 10,000 lignes** : Mode approximatif avec top-k = 20-50

---

## 💡 Exemples pratiques

### Exemple 1 : Comparaison simple sans LLM

**Objectif :** Comparer deux versions de spécifications

**Configuration :**
- Mode : En ligne
- Seuil : 0.78
- Analyse LLM : ❌ Désactivé
- Stratégie combinatoire : ❌ Désactivé

**Résultat :**
- Matches rapides basés uniquement sur la similarité
- Pas de validation sémantique
- Adapté pour des bases très similaires

### Exemple 2 : Comparaison avec validation LLM

**Objectif :** Détecter des équivalences sémantiques subtiles

**Configuration :**
- Mode : En ligne
- Seuil : 0.75 (plus souple)
- Analyse LLM : ✅ Activé (budget 300)
- Stratégie combinatoire : ❌ Désactivé

**Résultat :**
- Matches validés par IA
- Faux positifs filtrés
- Mismatches promus si équivalents
- Adapté pour des bases avec reformulations

### Exemple 3 : Comparaison avec recombinaison

**Objectif :** Matcher des exigences fragmentées

**Configuration :**
- Mode : En ligne
- Seuil : 0.78
- Analyse LLM : ✅ Auto-activé
- Stratégie combinatoire : ✅ Activé (max 4)

**Résultat :**
- Détection des exigences éclatées
- Recomposition automatique
- Validation LLM des combinaisons
- Adapté pour bases mal structurées

### Exemple 4 : Mode hors ligne complet

**Objectif :** Traitement autonome sans réseau

**Configuration :**
- Mode : 🔌 Hors ligne
- Modèle LLM : Qwen 2.5 3B
- Modèle embedding : BGE-M3
- Analyse LLM : ✅ Activé
- Stratégie combinatoire : ✅ Activé

**Résultat :**
- Traitement 100% local
- Plus lent que mode en ligne
- Fonctionne sans internet
- Adapté pour données sensibles

---

## ⚠️ Avertissements et limitations

### Matches combinatoires

**⚠️ Attention :** Un match combinatoire signifie qu'une ligne de Base 1 correspond à **plusieurs lignes** de Base 2.

**Implications :**
- La correspondance n'est pas 1-to-1
- Peut indiquer une granularité différente entre les bases
- Nécessite validation manuelle dans certains cas

**Validation :**
- Toujours analysés automatiquement par LLM
- Vérifier la colonne `warning` pour les détails
- Consulter `tgt_indices_combined` pour les lignes sources

### Performance

**Temps de traitement typiques :**
- 100 lignes : ~30 secondes (mode en ligne)
- 1,000 lignes : ~5 minutes
- 10,000 lignes : ~45 minutes

**Avec stratégie combinatoire :** +50% de temps

### Précision

La précision dépend de :
- Qualité des embeddings (modèle utilisé)
- Choix du seuil
- Activation de l'analyse LLM
- Cohérence des textes à comparer

**Recommandation :** Toujours valider manuellement un échantillon des résultats.

---

## 🆘 Résolution de problèmes

### Problème : Trop de faux positifs

**Solution :**
- Augmenter le seuil (0.82 - 0.85)
- Activer l'analyse LLM pour filtrer
- Vérifier que les colonnes comparées sont cohérentes

### Problème : Trop de faux négatifs

**Solution :**
- Diminuer le seuil (0.72 - 0.75)
- Activer la stratégie combinatoire
- Activer l'analyse LLM pour promouvoir

### Problème : Traitement très lent

**Solution :**
- Réduire la taille de batch (si problèmes mémoire)
- Utiliser mode approximatif (si > 10k lignes)
- Réduire le budget LLM
- Désactiver stratégie combinatoire (si non nécessaire)

### Problème : Matches combinatoires non pertinents

**Solution :**
- Réduire `max_combinations` (tester avec 2-3)
- Vérifier les résultats LLM (colonne `équivalence`)
- Augmenter le seuil global

---

## 📞 Support

Pour toute question ou problème :
- Consulter README.md et QUICKSTART_STREAMLIT.md
- Vérifier les logs de l'application
- Contacter l'équipe support

---

**Version du guide :** 1.0
**Dernière mise à jour :** 2025-11-22
