# 📊 CompareDB - Comparaison Sémantique Intelligente

Interface Streamlit moderne pour la comparaison sémantique de documents Excel avec support des modèles en ligne et hors ligne.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Proprietary-yellow.svg)](LICENSE)

---

## 🎯 Fonctionnalités

### Interface Streamlit moderne
- ✅ Interface utilisateur intuitive et responsive
- ✅ Upload/parcourir fichiers ou saisie manuelle
- ✅ Thème personnalisable (clair/sombre)
- ✅ Barre de progression en temps réel
- ✅ Export Excel direct

### Intelligence artificielle
- 🤖 **Analyse sémantique** avec embeddings
- 🔍 **Détection d'équivalences** via LLM
- 🔀 **Stratégie combinatoire** pour exigences fragmentées
- 📈 **Validation automatique** des matches

### Modes d'exécution
- 🌐 **Mode en ligne** : API Snowflake + DALLEM
- 🔌 **Mode hors ligne** : Modèles locaux (Qwen, Mistral, BGE-M3)
- 🔄 **Basculement simple** via toggle

---

## 🚀 Installation rapide

### Windows (2 clics)

```cmd
# 1. Installer
install.bat

# 2. Lancer
use.bat
```

### Linux/Mac

```bash
# Installer
pip install -r requirements.txt

# Lancer
streamlit run streamlit_app.py
```

**Interface** : http://localhost:8501

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**INSTALL_WINDOWS.md**](INSTALL_WINDOWS.md) | 🪟 Guide installation Windows complet |
| [**QUICKSTART_STREAMLIT.md**](QUICKSTART_STREAMLIT.md) | ⚡ Démarrage rapide (3 minutes) |
| [**README_STREAMLIT.md**](README_STREAMLIT.md) | 📖 Documentation Streamlit complète |
| [**USER_GUIDE.md**](USER_GUIDE.md) | 📘 Guide utilisateur avec schémas |

**Guide intégré** : Disponible dans l'interface (📘 dans la sidebar)

---

## 🎯 Utilisation

### 1. Configuration

**Sidebar > Configuration**
- Mode d'exécution : En ligne / Hors ligne
- Paramètres avancés : Seuil, batch size, mode matching
- Analyse LLM : Budget configurable
- Stratégie combinatoire : Max combinations

### 2. Fichiers

**Corps principal**
- Fichier 1 / Fichier 2
- Upload direct ou chemin manuel
- Sélection feuille + colonne

### 3. Résultats

**Après traitement**
- Métriques : Matches normaux, combinatoires, mismatches
- Tableaux interactifs avec tabs
- Export Excel automatique

---

## 🔧 Configuration

### Variables d'environnement (optionnel)

Créer un fichier `.env` :

```bash
# APIs en ligne
SNOWFLAKE_API_KEY=your_key
DALLEM_API_KEY=your_key

# Désactiver la vérification SSL (si nécessaire)
DISABLE_SSL_VERIFY=true
```

### Modèles hors ligne

Modifier `offline_models.py` pour les chemins locaux :

```python
AVAILABLE_LLM_MODELS = {
    "qwen": "C:\\Models\\Qwen\\Qwen2.5-3B-Instruct",
    "mistral": "C:\\Models\\mistralai\\Mistral-7B-Instruct-v0.3",
}

DEFAULT_EMBEDDING = "C:\\Models\\BAAI\\bge-m3"
```

**Télécharger les modèles** :
- [Qwen 2.5 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)

---

## 🔀 Stratégie combinatoire

**Innovation principale** : Détection automatique des exigences fragmentées

### Principe

Pour chaque mismatch :
1. Compare avec **toute la Base 2**
2. Sélectionne **top-k lignes** avec meilleurs scores
3. **Combine les textes** par concaténation
4. **Recalcule la similarité**
5. Si ≥ seuil → **Match combinatoire** ✅
6. Sinon → Essaie **k+1 lignes** (jusqu'à max)
7. **Validation LLM automatique**

### Exemple

```
Base 1 : "Le système doit supporter 1000 utilisateurs avec temps < 2s"

Base 2 fragmentée :
  [15] "Support de 1000 users"
  [31] "Performance < 2 secondes"
  [23] "Temps de réponse rapide"

→ Combinaison [15, 31, 23] : Score 0.81 ✅
→ Match combinatoire avec warning
→ LLM valide l'équivalence
```

---

## 📥 Résultats

### Fichiers Excel générés

#### `matches_YYYYMMDD_HHMMSS.xlsx`
Tous les matches (normaux + combinatoires)

**Colonnes principales** :
- `src_index`, `tgt_index` : Indices des lignes
- `source`, `target` : Textes comparés
- `score` : Similarité (0-1)
- `match_type` : "normal" / "combinatorial"
- `équivalence` : Validation LLM (TRUE/FALSE/None)
- `commentaire` : Explication LLM
- `analyse_llm` : Type d'analyse
- `tgt_indices_combined` : Indices combinés (si combinatoire)
- `warning` : Avertissement (si combinatoire)

#### `under_YYYYMMDD_HHMMSS.xlsx`
Mismatches définitifs (aucune correspondance)

---

## 🛠️ Développement

### Structure du projet

```
CompareDB/
├── streamlit_app.py          # Application principale
├── offline_models.py         # Support modèles locaux
├── requirements.txt          # Dépendances
│
├── install.bat               # Installation Windows
├── use.bat                   # Lancement Windows
├── run_streamlit.sh          # Lancement Linux/Mac
│
├── USER_GUIDE.md            # Guide utilisateur complet
├── INSTALL_WINDOWS.md       # Guide installation Windows
├── README_STREAMLIT.md      # Documentation Streamlit
├── QUICKSTART_STREAMLIT.md  # Démarrage rapide
│
├── .streamlit/
│   └── config.toml          # Configuration Streamlit
│
└── output/                  # Résultats Excel
```

### Technologies

- **Interface** : Streamlit 1.28+
- **IA** : OpenAI API / Transformers
- **Embeddings** : Snowflake Arctic / BGE-M3
- **LLM** : DALLEM / Qwen / Mistral
- **Data** : Pandas, NumPy, OpenPyXL

---

## ⚙️ Configuration système

### Minimum (mode en ligne)
- Python 3.10+
- RAM : 4 GB
- Disque : 500 MB

### Recommandé (mode hors ligne)
- Python 3.10+
- RAM : 16 GB
- Disque : 20 GB (modèles)
- GPU : NVIDIA avec CUDA (optionnel)

---

## 🐛 Dépannage

### Port 8501 occupé

```bash
# Linux/Mac
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Problème avec Streamlit

```bash
# Réinstaller
pip install --upgrade streamlit

# Tester
streamlit hello
```

### Mode hors ligne non disponible

```bash
# Installer les dépendances
pip install torch transformers sentence-transformers accelerate

# Vérifier
python -c "import torch; print(torch.__version__)"
```

---

## 📊 Exemples d'utilisation

### Cas 1 : Comparaison simple

**Configuration** :
- Mode : En ligne
- Analyse LLM : Désactivé
- Stratégie combinatoire : Désactivé

**Usage** : Comparaison rapide de deux versions de specs

### Cas 2 : Validation sémantique

**Configuration** :
- Mode : En ligne
- Analyse LLM : Activé (budget 300)
- Stratégie combinatoire : Désactivé

**Usage** : Détection d'équivalences subtiles avec validation IA

### Cas 3 : Exigences fragmentées

**Configuration** :
- Mode : En ligne
- Analyse LLM : Auto-activé
- Stratégie combinatoire : Activé (max 4)

**Usage** : Matching d'exigences éclatées en plusieurs lignes

### Cas 4 : Hors ligne complet

**Configuration** :
- Mode : Hors ligne
- Modèle : Qwen 2.5 3B + BGE-M3
- Analyse LLM : Activé
- Stratégie combinatoire : Activé

**Usage** : Traitement autonome sans réseau (données sensibles)

---

## 📄 Licence

**Propriétaire - Dassault Aviation**

Usage interne uniquement. Tous droits réservés.

---

## 🆘 Support

- **Documentation** : Voir [USER_GUIDE.md](USER_GUIDE.md)
- **Guide rapide** : Voir [QUICKSTART_STREAMLIT.md](QUICKSTART_STREAMLIT.md)
- **Installation Windows** : Voir [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md)

---

## 🎉 Démarrer maintenant

### Windows
```cmd
install.bat
use.bat
```

### Linux/Mac
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Accès** : http://localhost:8501

---

**Développé avec ❤️ pour Dassault Aviation**
