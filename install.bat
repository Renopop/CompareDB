@echo off
echo ========================================
echo CompareDB - Installation
echo ========================================
echo.

REM Verifier si Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n est pas installe ou n est pas dans le PATH
    echo.
    echo Veuillez installer Python 3.10 ou superieur depuis:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/4] Verification de Python...
python --version
echo.

echo [2/4] Mise a jour de pip...
python -m pip install --upgrade pip
echo.

echo [3/4] Installation des dependances de base...
pip install -r requirements.txt
echo.

echo [4/4] Creation du repertoire de sortie...
if not exist "output" mkdir output
echo.

echo ========================================
echo Installation terminee avec succes !
echo ========================================
echo.

REM Demander si l utilisateur veut installer les dependances pour le mode hors ligne
echo.
echo Voulez-vous installer les dependances pour le mode hors ligne ?
echo (Necessaire uniquement si vous voulez utiliser les modeles locaux)
echo.
echo Appuyez sur O pour Oui, N pour Non
choice /C ON /N /M "Votre choix (O/N): "

if errorlevel 2 (
    echo.
    echo Mode hors ligne ignore.
    echo Vous pourrez l installer plus tard avec:
    echo pip install torch transformers sentence-transformers accelerate
    goto :end
)

if errorlevel 1 (
    echo.
    echo [5/5] Installation des dependances pour le mode hors ligne...
    echo ATTENTION: Cette installation peut prendre plusieurs minutes.
    echo.
    pip install torch transformers sentence-transformers accelerate
    echo.
    echo Mode hors ligne installe !
)

:end
echo.
echo ========================================
echo.
echo Pour lancer l application:
echo   lancer.bat
echo.
echo Ou directement:
echo   streamlit run streamlit_app.py
echo.
echo ========================================
echo.
pause
