# Guide de démarrage rapide - CompareDB

## Installation rapide

### Windows

1. Double-cliquez sur `start.bat`
2. Ouvrez votre navigateur sur http://localhost:5000

### Linux/Mac

```bash
chmod +x start.sh
./start.sh
```

Ou manuellement :

```bash
pip install -r requirements.txt
python app.py
```

## Première utilisation

### Mode en ligne (par défaut)

1. Sélectionnez "Mode en ligne"
2. Renseignez les chemins des fichiers Excel
3. Cliquez sur "Lancer la comparaison"

### Mode hors ligne (avec modèles locaux)

#### Prérequis

1. Installez les dépendances supplémentaires :
   ```bash
   pip install torch transformers sentence-transformers accelerate
   ```

2. Téléchargez les modèles locaux :
   - **Qwen 2.5 3B** : https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
   - **Mistral 7B** : https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
   - **BGE-M3** : https://huggingface.co/BAAI/bge-m3

3. Modifiez les chemins dans `offline_models.py` :
   ```python
   AVAILABLE_LLM_MODELS = {
       "qwen": "votre/chemin/vers/Qwen2.5-3B-Instruct",
       "mistral": "votre/chemin/vers/Mistral-7B-Instruct-v0.3",
   }

   DEFAULT_EMBEDDING = "votre/chemin/vers/bge-m3"
   ```

#### Utilisation

1. Sélectionnez "Mode hors ligne"
2. Choisissez le modèle LLM et d'embedding
3. Renseignez les fichiers et lancez la comparaison

## Exemples d'utilisation

### Comparaison simple

**Fichiers** :
- Fichier 1 : `C:\Data\requirements_v1.xlsx`, Feuille : `Sheet1`, Colonne : 1
- Fichier 2 : `C:\Data\requirements_v2.xlsx`, Feuille : `Sheet1`, Colonne : 1

**Paramètres** :
- Seuil : 0.78 (par défaut)
- Mode : En ligne ou Hors ligne

### Comparaison avec analyse LLM

1. Cochez "Analyse LLM des équivalences"
2. Le système analysera les paires pour détecter les équivalences sémantiques
3. Les résultats incluront des colonnes supplémentaires :
   - `équivalence` : TRUE/FALSE
   - `commentaire` : Explication du LLM

### Mode approximatif (pour grandes bases)

1. Sélectionnez "Mode de matching" : Approximatif
2. Définissez le `Top-k` (par exemple : 10)
3. Plus rapide mais peut manquer certaines correspondances

## Thème sombre

Cliquez sur l'icône 🌙/☀️ en haut à droite pour basculer entre les thèmes clair et sombre.

## Résultats

Les fichiers de résultats sont générés dans le dossier `output/` :
- `matches_XXXXX.xlsx` : Paires au-dessus du seuil
- `under_XXXXX.xlsx` : Paires sous le seuil

## Dépannage

### Erreur "Mode hors ligne non disponible"

→ Installez les dépendances : `pip install torch transformers sentence-transformers`

### Erreur de lecture Excel

→ Vérifiez le chemin du fichier, le nom de la feuille et le numéro de colonne

### Le serveur ne démarre pas

→ Vérifiez que le port 5000 est disponible : `netstat -an | findstr 5000`

### Performance lente en mode hors ligne

→ Utilisez un GPU si disponible (CUDA), sinon réduisez la taille de batch

## Support

Pour plus d'informations, consultez le fichier README.md complet.
