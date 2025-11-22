#!/bin/bash

echo "========================================"
echo "CompareDB - Interface Web"
echo "========================================"
echo ""

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "ERREUR: Python 3 n'est pas installé"
    exit 1
fi

echo "[1/3] Vérification des dépendances..."
pip3 install -q -r requirements.txt

echo "[2/3] Création du répertoire de sortie..."
mkdir -p output

echo "[3/3] Démarrage du serveur..."
echo ""
echo "Interface disponible sur: http://localhost:5000"
echo "Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""
echo "========================================"
echo ""

python3 app.py
