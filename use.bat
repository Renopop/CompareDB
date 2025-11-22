@echo off
echo ========================================
echo CompareDB - Interface Streamlit
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installé ou n'est pas dans le PATH
    echo.
    echo Veuillez d'abord exécuter install.bat
    echo.
    pause
    exit /b 1
)

REM Vérifier si Streamlit est installé
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Streamlit n'est pas installé
    echo.
    echo Veuillez d'abord exécuter install.bat
    echo.
    pause
    exit /b 1
)

echo [1/2] Vérification de l'installation...
echo Python: OK
echo Streamlit: OK
echo.

echo [2/2] Démarrage de l'interface Streamlit...
echo.
echo L'interface va s'ouvrir automatiquement dans votre navigateur.
echo URL: http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arrêter le serveur
echo.
echo ========================================
echo.

REM Lancer Streamlit
streamlit run streamlit_app.py

REM Si Streamlit s'arrête avec une erreur
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERREUR lors du démarrage de Streamlit
    echo ========================================
    echo.
    echo Solutions possibles:
    echo 1. Vérifier que le port 8501 n'est pas déjà utilisé
    echo 2. Réinstaller avec install.bat
    echo 3. Lancer manuellement: streamlit run streamlit_app.py
    echo.
    pause
    exit /b 1
)
