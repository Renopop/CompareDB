# CompareDB - Interface Streamlit 🚀

Interface Streamlit moderne pour la comparaison sémantique de documents Excel avec support des modèles en ligne et hors ligne.

## 🎯 Fonctionnalités

- ✨ **Interface Streamlit moderne** et intuitive
- 🔌 **Toggle hors ligne** : Basculez facilement entre mode en ligne et hors ligne
- 🤖 **Modèles locaux** : Qwen 2.5 3B, Mistral 7B, BGE-M3
- 🌐 **API distantes** : Support des API Snowflake et DALLEM
- 📊 **Visualisation en temps réel** des résultats
- 💾 **Export Excel** direct avec bouton de téléchargement
- 🎨 **Thème personnalisable** (clair/sombre automatique)

## 🚀 Démarrage rapide

### Commande simple

```bash
streamlit run streamlit_app.py
```

### Avec les scripts

**Windows** :
```bash
run_streamlit.bat
```

**Linux/Mac** :
```bash
chmod +x run_streamlit.sh
./run_streamlit.sh
```

L'interface s'ouvrira automatiquement sur : **http://localhost:8501**

## 📦 Installation

### 1. Installer les dépendances de base

```bash
pip install -r requirements.txt
```

### 2. Mode hors ligne (optionnel)

Pour utiliser les modèles locaux :

```bash
pip install torch transformers sentence-transformers accelerate
```

### 3. Configurer les modèles locaux

Modifiez les chemins dans `offline_models.py` :

```python
AVAILABLE_LLM_MODELS = {
    "qwen": "D:\\IA Test\\models\\Qwen\\Qwen2.5-3B-Instruct",
    "mistral": "D:\\IA Test\\models\\mistralai\\Mistral-7B-Instruct-v0.3",
}

DEFAULT_EMBEDDING = "D:\\IA Test\\models\\BAAI\\bge-m3"
```

## 🎮 Utilisation

### 1. Lancer l'application

```bash
streamlit run streamlit_app.py
```

### 2. Configurer l'interface

#### Mode d'exécution (Sidebar)

- **🔌 Mode hors ligne** : Toggle ON pour utiliser les modèles locaux
  - Sélectionner le modèle LLM : Qwen ou Mistral
  - Sélectionner le modèle d'embedding : BGE-M3

- **🌐 Mode en ligne** : Toggle OFF pour utiliser les API

#### Fichiers à comparer

- **Fichier 1** : Chemin, nom de feuille, numéro de colonne
- **Fichier 2** : Chemin, nom de feuille, numéro de colonne

#### Paramètres avancés (expandable)

- **Seuil de similarité** : 0.0 à 1.0 (défaut: 0.78)
- **Taille de batch** : Nombre d'éléments traités simultanément
- **Limite de lignes** : Pour tester avec moins de données
- **Mode de matching** : Complet ou Approximatif (top-k)
- **Analyse LLM** : Active l'analyse sémantique par LLM

### 3. Lancer la comparaison

Cliquez sur **"▶️ Lancer la comparaison"**

### 4. Consulter les résultats

L'application affiche :
- 📊 **Métriques** : Nombre de matches, sous le seuil, taux de match
- 📋 **Tableaux** : Aperçu des correspondances et non-correspondances
- 📥 **Téléchargements** : Boutons pour télécharger les fichiers Excel

## 🎨 Interface

### Barre latérale (Sidebar)

```
⚙️ Configuration
├── Mode d'exécution
│   └── 🔌 Toggle Mode hors ligne
├── Modèles locaux (si hors ligne)
│   ├── Modèle LLM
│   └── Modèle d'embedding
└── 🔧 Paramètres avancés
    ├── Seuil de similarité
    ├── Taille de batch
    ├── Limite de lignes
    ├── Mode de matching
    └── Analyse LLM
```

### Corps principal

```
📊 CompareDB
├── 📁 Fichier 1 | 📁 Fichier 2
├── ▶️ Lancer la comparaison
├── 🎯 Résultats
│   ├── Métriques (3 colonnes)
│   ├── Tabs : Matches / Sous le seuil
│   └── 📥 Téléchargements
└── Footer (informations mode/modèles)
```

## 🎯 Exemples d'utilisation

### Exemple 1 : Comparaison simple en ligne

1. Laisser le toggle **Mode hors ligne** sur OFF
2. Renseigner les fichiers :
   - Fichier 1 : `C:\Data\requirements_v1.xlsx`, Feuille : `Sheet1`, Colonne : `1`
   - Fichier 2 : `C:\Data\requirements_v2.xlsx`, Feuille : `Sheet1`, Colonne : `1`
3. Cliquer sur **"▶️ Lancer la comparaison"**
4. Télécharger les résultats

### Exemple 2 : Comparaison hors ligne avec LLM

1. Activer le toggle **🔌 Mode hors ligne**
2. Sélectionner :
   - Modèle LLM : **Qwen 2.5 3B**
   - Modèle d'embedding : **BGE-M3**
3. Dans **Paramètres avancés** :
   - Cocher **Analyse LLM des équivalences**
4. Renseigner les fichiers et lancer
5. Les résultats incluront les colonnes `équivalence` et `commentaire`

### Exemple 3 : Mode approximatif pour grandes bases

1. Dans **Paramètres avancés** :
   - Mode de matching : **Approximatif (top-k)**
   - Top-k : `10`
2. Lancer la comparaison
3. Traitement plus rapide avec légère perte de précision

## 🔧 Configuration Streamlit

Le fichier `.streamlit/config.toml` permet de personnaliser :

```toml
[theme]
primaryColor = "#4f46e5"        # Couleur principale
backgroundColor = "#ffffff"      # Fond
secondaryBackgroundColor = "#f9fafb"  # Fond secondaire
textColor = "#111827"           # Texte

[server]
port = 8501                     # Port du serveur
```

## 📱 Responsive Design

L'interface s'adapte automatiquement :
- **Desktop** : 2 colonnes pour les fichiers
- **Tablet/Mobile** : 1 colonne, layout vertical

## 🐛 Dépannage

### Erreur "Mode hors ligne non disponible"

```bash
pip install torch transformers sentence-transformers accelerate
```

### Port 8501 déjà utilisé

Modifier le port dans `.streamlit/config.toml` :
```toml
[server]
port = 8502
```

Ou lancer avec :
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Streamlit ne démarre pas

Vérifier l'installation :
```bash
pip install --upgrade streamlit
streamlit hello  # Test de Streamlit
```

### Erreur de lecture des modèles locaux

Vérifier les chemins dans `offline_models.py` et que les modèles sont téléchargés.

## 🆚 Flask vs Streamlit

| Caractéristique | Flask | Streamlit |
|----------------|-------|-----------|
| Interface | HTML/CSS/JS custom | Widgets Python natifs |
| Développement | Plus long | Très rapide |
| Personnalisation | Totale | Limitée au thème |
| Performance | Meilleure | Bonne |
| Déploiement | Standard | Streamlit Cloud |
| **Recommandation** | Production | Prototypage/Interne |

## 📚 Ressources

- [Documentation Streamlit](https://docs.streamlit.io)
- [Composants Streamlit](https://streamlit.io/components)
- [Galerie d'apps](https://streamlit.io/gallery)

## 🚀 Déploiement

### Streamlit Cloud (gratuit)

1. Pusher le code sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter le repository
4. Déployer en 1 clic

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📄 Licence

Propriétaire - Dassault Aviation

## 💡 Support

Pour toute question, consultez la documentation ou contactez l'équipe.

---

**Note** : Cette version Streamlit est plus simple à utiliser et à maintenir que la version Flask. Elle est recommandée pour un usage interne et des prototypes rapides.
