@echo off
echo ========================================
echo    CompareDB - Lancement
echo ========================================
echo.

REM Verifier si Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n est pas installe ou n est pas dans le PATH
    echo.
    echo Veuillez d abord executer install.bat
    echo.
    pause
    exit /b 1
)

REM Verifier si Streamlit est installe
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Streamlit n est pas installe
    echo.
    echo Veuillez d abord executer install.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte
echo [OK] Streamlit detecte
echo.

REM Creer le repertoire output si necessaire
if not exist "output" mkdir output

echo Demarrage de l interface...
echo Le navigateur va s ouvrir automatiquement.
echo.
echo ========================================
echo   Pour arreter : Ctrl+C
echo ========================================
echo.

REM Lancer Streamlit avec ouverture automatique du navigateur
streamlit run streamlit_app.py --server.headless false --browser.gatherUsageStats false

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERREUR lors du demarrage
    echo ========================================
    echo.
    echo Solutions possibles:
    echo 1. Verifier que le port 8501 n est pas utilise
    echo 2. Reinstaller avec install.bat
    echo.
    pause
)
