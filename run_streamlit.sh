#!/bin/bash

echo "========================================"
echo "CompareDB - Interface Streamlit"
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

echo "[3/3] Démarrage de Streamlit..."
echo ""
echo "Interface disponible sur: http://localhost:8501"
echo "Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""
echo "========================================"
echo ""

streamlit run streamlit_app.py
