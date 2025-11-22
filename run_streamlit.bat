@echo off
echo ========================================
echo CompareDB - Interface Streamlit
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installé ou n'est pas dans le PATH
    pause
    exit /b 1
)

echo [1/3] Vérification des dépendances...
pip install -q -r requirements.txt

echo [2/3] Création du répertoire de sortie...
if not exist output mkdir output

echo [3/3] Démarrage de Streamlit...
echo.
echo Interface disponible sur: http://localhost:8501
echo Appuyez sur Ctrl+C pour arrêter le serveur
echo.
echo ========================================
echo.

streamlit run streamlit_app.py

pause
