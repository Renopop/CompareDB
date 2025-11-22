# 🪟 Installation Windows - CompareDB

## 🚀 Installation rapide (2 étapes)

### 1️⃣ Installer

Double-cliquez sur **`install.bat`**

Le script va :
- ✅ Vérifier Python
- ✅ Installer toutes les dépendances
- ✅ Créer les répertoires nécessaires
- ❓ Demander si vous voulez le mode hors ligne (optionnel)

### 2️⃣ Lancer

Double-cliquez sur **`use.bat`**

L'interface s'ouvre automatiquement sur : **http://localhost:8501**

---

## 📋 Prérequis

**Python 3.10 ou supérieur** doit être installé.

Télécharger Python : https://www.python.org/downloads/

⚠️ **Lors de l'installation de Python, cochez "Add Python to PATH"**

---

## 🔧 Installation détaillée

### Vérifier si Python est installé

Ouvrir **PowerShell** ou **Invite de commandes** :

```cmd
python --version
```

Doit afficher : `Python 3.10.x` ou supérieur

### Option 1 : Installation automatique (recommandé)

```cmd
install.bat
```

### Option 2 : Installation manuelle

```cmd
# Installer les dépendances de base
pip install -r requirements.txt

# (Optionnel) Mode hors ligne
pip install torch transformers sentence-transformers accelerate
```

---

## ▶️ Utilisation

### Option 1 : Avec le script (recommandé)

```cmd
use.bat
```

### Option 2 : Commande directe

```cmd
streamlit run streamlit_app.py
```

### Option 3 : Avec le script existant

```cmd
run_streamlit.bat
```

---

## 🔌 Mode hors ligne

Le mode hors ligne permet d'utiliser des modèles locaux sans connexion internet.

### Installation

Lors de l'exécution de `install.bat`, répondre **O** (Oui) quand demandé.

Ou manuellement :

```cmd
pip install torch transformers sentence-transformers accelerate
```

### Configuration

Modifier les chemins des modèles dans `offline_models.py` :

```python
AVAILABLE_LLM_MODELS = {
    "qwen": "C:\\Models\\Qwen\\Qwen2.5-3B-Instruct",
    "mistral": "C:\\Models\\mistralai\\Mistral-7B-Instruct-v0.3",
}

DEFAULT_EMBEDDING = "C:\\Models\\BAAI\\bge-m3"
```

### Télécharger les modèles

Les modèles doivent être téléchargés depuis Hugging Face :

- **Qwen 2.5 3B** : https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- **Mistral 7B** : https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- **BGE-M3** : https://huggingface.co/BAAI/bge-m3

---

## ⚙️ Configuration système requise

### Minimum (mode en ligne)
- **Python** : 3.10+
- **RAM** : 4 GB
- **Disque** : 500 MB

### Recommandé (mode hors ligne)
- **Python** : 3.10+
- **RAM** : 16 GB
- **Disque** : 20 GB (pour les modèles)
- **GPU** : Optionnel mais recommandé (NVIDIA avec CUDA)

---

## 🐛 Dépannage

### Erreur : Python n'est pas reconnu

**Solution** : Ajouter Python au PATH

1. Ouvrir les **Variables d'environnement**
2. Ajouter le chemin Python (ex: `C:\Python310`)
3. Redémarrer l'invite de commandes

### Erreur : Port 8501 déjà utilisé

**Solution 1** : Tuer le processus existant

```cmd
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

**Solution 2** : Utiliser un autre port

```cmd
streamlit run streamlit_app.py --server.port 8502
```

### Erreur : Module 'streamlit' not found

**Solution** : Réinstaller

```cmd
pip install --upgrade streamlit
```

### Problème avec le mode hors ligne

**Solution** : Vérifier l'installation

```cmd
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

Si erreur, réinstaller :

```cmd
pip install --upgrade torch transformers sentence-transformers accelerate
```

---

## 📁 Structure des fichiers

```
CompareDB/
├── install.bat              ← Installer les dépendances
├── use.bat                  ← Lancer l'application
├── run_streamlit.bat        ← Alternative de lancement
├── streamlit_app.py         ← Application principale
├── offline_models.py        ← Configuration modèles locaux
├── requirements.txt         ← Liste des dépendances
├── USER_GUIDE.md           ← Guide utilisateur complet
├── .streamlit/
│   └── config.toml         ← Configuration Streamlit
└── output/                 ← Résultats générés (xlsx)
```

---

## 📚 Documentation

- **Guide utilisateur** : Accessible dans l'interface (📘 dans la sidebar)
- **Guide rapide** : `QUICKSTART_STREAMLIT.md`
- **Documentation Streamlit** : `README_STREAMLIT.md`
- **Installation Linux/Mac** : Voir `QUICKSTART_STREAMLIT.md`

---

## ✅ Vérification de l'installation

Après avoir exécuté `install.bat`, vérifier :

```cmd
python --version
pip show streamlit
pip show pandas
pip show numpy
```

Tous devraient afficher des versions valides.

---

## 🆘 Support

En cas de problème :

1. Vérifier les prérequis (Python 3.10+)
2. Réinstaller avec `install.bat`
3. Consulter le guide de dépannage ci-dessus
4. Vérifier les logs dans la console

---

**Bonne utilisation de CompareDB !** 🎉
