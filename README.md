# CompareDB - Comparaison Sémantique Intelligente

Interface web moderne pour la comparaison sémantique de documents Excel avec support des modèles en ligne et hors ligne.

## Fonctionnalités

- **Interface moderne** avec thème clair/sombre
- **Mode en ligne** : utilise des API distantes pour les embeddings et LLM
- **Mode hors ligne** : utilise des modèles locaux (Qwen, Mistral, BGE-M3)
- **Analyse sémantique** avancée avec embeddings
- **Détection d'équivalences** via LLM
- **Export Excel** des résultats

## Installation

### Dépendances de base

```bash
pip install -r requirements.txt
```

### Mode hors ligne (optionnel)

Pour activer le mode hors ligne avec les modèles locaux :

```bash
pip install torch transformers sentence-transformers accelerate
```

**Note** : Les modèles locaux doivent être téléchargés aux chemins suivants :
- `D:\IA Test\models\Qwen\Qwen2.5-3B-Instruct`
- `D:\IA Test\models\mistralai\Mistral-7B-Instruct-v0.3`
- `D:\IA Test\models\BAAI\bge-m3`

Vous pouvez modifier ces chemins dans le fichier `offline_models.py`.

## Utilisation

### Démarrer le serveur

```bash
python app.py
```

L'interface sera accessible sur : http://localhost:5000

### Utilisation de l'interface

1. **Sélectionner le mode** : En ligne (API) ou Hors ligne (modèles locaux)
2. **Configurer les fichiers** :
   - Chemin du fichier Excel 1
   - Nom de la feuille et numéro de colonne
   - Même chose pour le fichier 2
3. **Ajuster les options avancées** (optionnel) :
   - Seuil de similarité (0.78 par défaut)
   - Taille de batch pour les embeddings
   - Activation de l'analyse LLM
   - Mode de matching (complet ou approximatif)
4. **Lancer la comparaison**
5. **Télécharger les résultats** :
   - `matches.xlsx` : paires au-dessus du seuil
   - `under_threshold.xlsx` : paires sous le seuil

## Configuration

### Mode en ligne

Les variables d'environnement suivantes peuvent être utilisées :

```bash
export SNOWFLAKE_API_BASE="https://api.example.com/snowflake/v1"
export SNOWFLAKE_API_KEY="votre_clé"
export DALLEM_API_BASE="https://api.example.com/dallem/v1"
export DALLEM_API_KEY="votre_clé"
export DISABLE_SSL_VERIFY="false"
```

### Mode hors ligne

Modifiez les chemins des modèles dans `offline_models.py` :

```python
AVAILABLE_LLM_MODELS = {
    "qwen": "chemin/vers/Qwen2.5-3B-Instruct",
    "mistral": "chemin/vers/Mistral-7B-Instruct-v0.3",
}

DEFAULT_EMBEDDING = "chemin/vers/bge-m3"
```

## Structure du projet

```
CompareDB/
├── app.py                  # Serveur Flask
├── test2_v4.py            # Logique de comparaison sémantique
├── offline_models.py      # Gestion des modèles locaux
├── requirements.txt       # Dépendances Python
├── static/               # Interface web
│   ├── index.html        # Page principale
│   ├── styles.css        # Styles (avec dark mode)
│   └── app.js            # Logique frontend
└── output/               # Résultats générés
```

## Utilisation en ligne de commande (legacy)

L'ancien mode CLI est toujours disponible :

```bash
python test2_v4.py --interactive
```

Ou avec des arguments :

```bash
python test2_v4.py \
  --file1 "fichier1.xlsx" --sheet1 "Feuil1" --col1 1 \
  --file2 "fichier2.xlsx" --sheet2 "Feuil1" --col2 1 \
  --threshold 0.78 \
  --llm-equivalent
```

## Caractéristiques techniques

- **Matching global 2 phases** : Assure l'unicité des paires (1 source → 1 cible)
- **Support du top-k approximatif** : Pour de très grandes bases
- **Normalisation des embeddings** : Cosinus similarity optimisée
- **Gestion des erreurs robuste** : Retry automatique sur les erreurs réseau
- **Interface responsive** : Fonctionne sur desktop et mobile
- **Thème adaptatif** : Mode clair/sombre automatique

## Licence

Propriétaire - Dassault Aviation

## Support

Pour toute question ou problème, consultez la documentation ou contactez l'équipe de support.
