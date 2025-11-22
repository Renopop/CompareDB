# 🚀 Guide de démarrage rapide - CompareDB Streamlit

## Installation en 3 étapes

### 1️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2️⃣ Lancer l'application

**Windows** :
```bash
run_streamlit.bat
```

**Linux/Mac** :
```bash
./run_streamlit.sh
```

**Ou directement** :
```bash
streamlit run streamlit_app.py
```

### 3️⃣ Ouvrir l'interface

L'application s'ouvre automatiquement sur : **http://localhost:8501**

---

## 🎮 Utilisation rapide

### Mode en ligne (par défaut)

1. Laisser le toggle **"Mode hors ligne"** sur **OFF**
2. Remplir les champs :
   - **Fichier 1** : Chemin Excel + Feuille + Colonne
   - **Fichier 2** : Chemin Excel + Feuille + Colonne
3. Cliquer sur **"▶️ Lancer la comparaison"**
4. Télécharger les résultats avec les boutons 📥

### Mode hors ligne (modèles locaux)

1. **Activer** le toggle **"🔌 Mode hors ligne"** dans la barre latérale
2. Sélectionner :
   - **Modèle LLM** : Qwen 2.5 3B ou Mistral 7B
   - **Modèle d'embedding** : BGE-M3
3. Remplir les fichiers et lancer

⚠️ **Prérequis pour le mode hors ligne** :

```bash
# Installer les dépendances
pip install torch transformers sentence-transformers accelerate

# Modifier les chemins des modèles dans offline_models.py
```

---

## 📊 Interface

```
┌─────────────────────────────────────────┐
│  📊 CompareDB                           │
│  Comparaison sémantique intelligente    │
├─────────────────────────────────────────┤
│                                         │
│  [Fichier 1]    [Fichier 2]            │
│                                         │
│  ▶️ Lancer la comparaison               │
│                                         │
│  📊 Résultats                           │
│  ├── Métriques                          │
│  ├── Tableaux                           │
│  └── 📥 Téléchargements                 │
│                                         │
└─────────────────────────────────────────┘

Sidebar (gauche):
├── ⚙️ Configuration
├── 🔌 Toggle Mode hors ligne
├── Sélection modèles (si hors ligne)
└── 🔧 Paramètres avancés
```

---

## 🎯 Commande PyCharm

Dans le terminal PyCharm (`Alt + F12`) :

```bash
streamlit run streamlit_app.py
```

---

## ⚙️ Options avancées

Ouvrir **"🔧 Paramètres avancés"** dans la sidebar :

- **Seuil de similarité** : 0.0 - 1.0 (défaut: 0.78)
- **Taille de batch** : Nombre d'éléments par batch (défaut: 16)
- **Limite de lignes** : Tester avec moins de données
- **Mode de matching** :
  - **Complet** : Matrice complète (précis)
  - **Approximatif** : Top-k (rapide)
- **Analyse LLM** : Détection d'équivalences sémantiques

---

## 📥 Résultats

Après traitement, vous obtenez :

### Métriques (3 cartes)
- ✅ **Matches** : Nombre de correspondances
- ⚠️ **Sous le seuil** : Non-correspondances
- 📊 **Taux de match** : Pourcentage

### Tableaux interactifs
- **Tab "Matches"** : Correspondances ≥ seuil
- **Tab "Sous le seuil"** : Correspondances < seuil

### Fichiers Excel
- 📥 `matches_YYYYMMDD_HHMMSS.xlsx`
- 📥 `under_YYYYMMDD_HHMMSS.xlsx`

---

## 🎨 Thème

Streamlit détecte automatiquement votre préférence système (clair/sombre).

**Personnaliser le thème** :

Éditer `.streamlit/config.toml` :

```toml
[theme]
primaryColor = "#4f46e5"        # Violet
backgroundColor = "#ffffff"      # Blanc
textColor = "#111827"           # Noir
```

**Basculer manuellement** :

`☰ Menu` (en haut à droite) → `Settings` → `Theme`

---

## 💡 Astuces

### Raccourcis Streamlit
- `R` : Relancer l'application
- `C` : Effacer le cache

### Performance
- **Grandes bases** : Utilisez le mode "Approximatif"
- **GPU** : Streamlit détecte automatiquement CUDA pour les modèles locaux
- **Batch size** : Augmentez pour GPU, diminuez pour CPU

### Sauvegarde
Les fichiers sont sauvegardés dans `output/` avec timestamp unique.

---

## ❓ Problèmes fréquents

| Problème | Solution |
|----------|----------|
| Mode hors ligne non dispo | `pip install torch transformers sentence-transformers` |
| Port 8501 occupé | `streamlit run streamlit_app.py --server.port 8502` |
| Erreur Excel | Vérifier chemin, feuille et colonne |
| Lenteur | Réduire batch size ou activer mode approximatif |

---

## 🚀 Prêt à démarrer !

```bash
# Commande unique
streamlit run streamlit_app.py
```

Puis ouvrez : **http://localhost:8501** 🎉
