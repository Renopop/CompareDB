@echo off
echo ========================================
echo CompareDB - Installation
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installé ou n'est pas dans le PATH
    echo.
    echo Veuillez installer Python 3.10 ou supérieur depuis:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/4] Vérification de Python...
python --version
echo.

echo [2/4] Mise à jour de pip...
python -m pip install --upgrade pip
echo.

echo [3/4] Installation des dépendances de base...
pip install -r requirements.txt
echo.

echo [4/4] Création du répertoire de sortie...
if not exist "output" mkdir output
echo.

echo ========================================
echo Installation terminée avec succès !
echo ========================================
echo.

REM Demander si l'utilisateur veut installer les dépendances pour le mode hors ligne
echo.
echo Voulez-vous installer les dépendances pour le mode hors ligne ?
echo (Nécessaire uniquement si vous voulez utiliser les modèles locaux)
echo.
echo Appuyez sur O pour Oui, N pour Non
choice /C ON /N /M "Votre choix (O/N): "

if errorlevel 2 (
    echo.
    echo Mode hors ligne ignoré.
    echo Vous pourrez l'installer plus tard en décommentant les lignes dans requirements.txt
    echo et en exécutant: pip install torch transformers sentence-transformers accelerate
    goto :end
)

if errorlevel 1 (
    echo.
    echo [5/5] Installation des dépendances pour le mode hors ligne...
    echo ATTENTION: Cette installation peut prendre plusieurs minutes et nécessite plusieurs Go d'espace disque.
    echo.
    pip install torch transformers sentence-transformers accelerate
    echo.
    echo Mode hors ligne installé !
)

:end
echo.
echo ========================================
echo.
echo Pour lancer l'application:
echo   lancer.bat
echo.
echo Ou directement:
echo   streamlit run streamlit_app.py
echo.
echo ========================================
echo.
pause
