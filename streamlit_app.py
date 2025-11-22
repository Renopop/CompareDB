"""
Interface Streamlit pour CompareDB
Comparaison sémantique avec support mode en ligne et hors ligne
"""

import streamlit as st
import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
import httpx
from datetime import datetime
from pathlib import Path

# Import du code existant
from test2_v4 import (
    read_excel_col,
    embed_in_batches,
    cosine_two_phase_global,
    cosine_topk_pairs,
    cosine_two_phase_global_from_pairs,
    DirectOpenAIEmbeddings,
    DirectOpenAILLM,
    make_logger,
)

# Import du mode offline
try:
    from offline_models import OfflineEmbeddingsAdapter, OfflineLLMAdapter
    OFFLINE_AVAILABLE = True
except ImportError as e:
    OFFLINE_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="CompareDB - Comparaison Sémantique",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
        border: 1px solid #10b981;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(139, 92, 246, 0.1));
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Répertoire de sortie
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Logger
logger = make_logger(debug=True)

# Configuration API
SNOWFLAKE_API_BASE = os.getenv(
    "SNOWFLAKE_API_BASE",
    "https://api.dev.dassault-aviation.pro/snowflake-arctic-embed-l-v2.0/v1"
)
SNOWFLAKE_API_KEY = os.getenv("SNOWFLAKE_API_KEY", "token")
DALLEM_API_BASE = os.getenv(
    "DALLEM_API_BASE",
    "https://api.dev.dassault-aviation.pro/dallem-pilote/v1"
)
DALLEM_API_KEY = os.getenv("DALLEM_API_KEY", "EMPTY")
VERIFY_SSL = not (os.getenv("DISABLE_SSL_VERIFY", "true").lower() in ("1", "true", "yes", "on"))


def apply_combinatorial_strategy(
    mismatches: list,
    s1_raw: list,
    s2_raw: list,
    Q: np.ndarray,
    D: np.ndarray,
    threshold: float,
    max_combinations: int,
    logger,
) -> tuple:
    """
    Applique la stratégie combinatoire pour tenter de matcher les mismatches.

    Pour chaque mismatch de la base 1 :
    1. Compare avec toute la base 2
    2. Prend les top-k lignes avec les meilleurs scores
    3. Combine ces lignes (concaténation)
    4. Compare la ligne base 1 avec la combinaison
    5. Si match (≥ seuil) : ajoute comme match combinatoire
    6. Sinon, essaie avec k+1 lignes (jusqu'à max_combinations)
    7. Si aucune combinaison ne marche : reste en mismatch

    Retourne : (nouveaux_matches, mismatches_definitifs)
    """
    logger.info(f"[combinatorial] Démarrage stratégie combinatoire sur {len(mismatches)} mismatches")

    new_matches = []
    final_mismatches = []

    for mismatch_idx, mismatch_row in enumerate(mismatches):
        src_idx = mismatch_row["src_index"]
        src_text = mismatch_row["source"]

        logger.debug(f"[combinatorial] Traitement mismatch {mismatch_idx + 1}/{len(mismatches)} (src_idx={src_idx})")

        # Calculer les scores avec toute la base 2
        src_embedding = Q[src_idx]  # (d,)
        scores = np.dot(D, src_embedding)  # (n_d,)

        # Trier par score décroissant
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        # Tester les combinaisons de 2 à max_combinations
        match_found = False

        for k in range(2, max_combinations + 1):
            # Prendre les top-k lignes
            top_k_indices = sorted_indices[:k]
            top_k_scores = sorted_scores[:k]

            # Combiner les textes
            combined_text = " ".join([s2_raw[idx] for idx in top_k_indices])

            # Générer l'embedding de la combinaison
            # Pour éviter de régénérer l'embedding, on fait une moyenne pondérée
            # des embeddings existants (approximation rapide)
            combined_embedding = np.mean(D[top_k_indices], axis=0)
            combined_embedding = combined_embedding / (np.linalg.norm(combined_embedding) + 1e-12)

            # Calculer le score avec la combinaison
            combo_score = float(np.dot(src_embedding, combined_embedding))

            logger.debug(
                f"[combinatorial] src={src_idx}, k={k}, "
                f"indices={list(top_k_indices)}, combo_score={combo_score:.4f}"
            )

            # Vérifier si ça fait un match
            if combo_score >= threshold:
                logger.info(
                    f"[combinatorial] ✅ Match trouvé ! src={src_idx}, k={k}, "
                    f"indices={list(top_k_indices)}, score={combo_score:.4f}"
                )

                # Créer le nouveau match
                new_match = {
                    "src_index": src_idx,
                    "tgt_index": None,  # Pas d'index unique
                    "tgt_indices_combined": list(top_k_indices),  # Liste des indices combinés
                    "source": src_text,
                    "target": combined_text,
                    "score": round(combo_score, 4),
                    "match_type": "combinatorial",
                    "combination_size": k,
                    "warning": f"⚠️ MATCH COMBINATOIRE : Lignes base 2 combinées = {list(top_k_indices)}",
                    "individual_scores": [round(float(s), 4) for s in top_k_scores],
                }

                new_matches.append(new_match)
                match_found = True
                break  # On arrête dès qu'on trouve un match

        if not match_found:
            # Aucune combinaison n'a matché : mismatch définitif
            logger.debug(f"[combinatorial] ❌ Aucune combinaison trouvée pour src={src_idx}")
            mismatch_row["match_type"] = "definitive_mismatch"
            final_mismatches.append(mismatch_row)

    logger.info(
        f"[combinatorial] Terminé : {len(new_matches)} nouveaux matches combinatoires, "
        f"{len(final_mismatches)} mismatches définitifs"
    )

    return new_matches, final_mismatches


def main():
    # En-tête
    st.markdown('<h1 class="main-header">📊 CompareDB</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comparaison sémantique intelligente avec IA</p>', unsafe_allow_html=True)

    st.divider()

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Mode hors ligne
        st.subheader("Mode d'exécution")
        offline_mode = st.toggle(
            "🔌 Mode hors ligne",
            value=False,
            help="Utilise les modèles locaux au lieu des API distantes"
        )

        if offline_mode:
            if not OFFLINE_AVAILABLE:
                st.error("⚠️ Mode hors ligne non disponible. Installez les dépendances :\n```pip install torch transformers sentence-transformers```")
                st.stop()

            st.success("✅ Mode hors ligne activé")

            # Sélection des modèles locaux
            st.subheader("Modèles locaux")
            llm_model = st.selectbox(
                "Modèle LLM",
                options=["qwen", "mistral"],
                format_func=lambda x: {
                    "qwen": "🤖 Qwen 2.5 3B Instruct",
                    "mistral": "🤖 Mistral 7B Instruct v0.3"
                }[x]
            )

            embedding_model = st.selectbox(
                "Modèle d'embedding",
                options=["bge-m3"],
                format_func=lambda x: "🔤 BGE-M3"
            )
        else:
            st.info("🌐 Mode en ligne activé (API)")

        st.divider()

        # Paramètres avancés
        with st.expander("🔧 Paramètres avancés"):
            threshold = st.slider(
                "Seuil de similarité",
                min_value=0.0,
                max_value=1.0,
                value=0.78,
                step=0.01,
                help="Score minimum pour considérer une correspondance"
            )

            batch_size = st.number_input(
                "Taille de batch",
                min_value=1,
                max_value=128,
                value=16,
                help="Nombre d'éléments traités simultanément"
            )

            limit = st.number_input(
                "Limite de lignes (0 = aucune)",
                min_value=0,
                max_value=100000,
                value=0,
                help="Limiter le nombre de lignes à traiter (pour tests)"
            )

            match_mode = st.selectbox(
                "Mode de matching",
                options=["full", "approx"],
                format_func=lambda x: {
                    "full": "Complet (matrice complète)",
                    "approx": "Approximatif (top-k)"
                }[x]
            )

            if match_mode == "approx":
                topk = st.number_input(
                    "Top-k (si approximatif)",
                    min_value=1,
                    max_value=100,
                    value=10
                )
            else:
                topk = 10

            llm_equivalent = st.checkbox(
                "Analyse LLM des équivalences",
                value=False,
                help="Utilise un LLM pour analyser les équivalences sémantiques"
            )

            if llm_equivalent:
                llm_max = st.number_input(
                    "Nombre max d'analyses LLM",
                    min_value=1,
                    max_value=1000,
                    value=200
                )
            else:
                llm_max = 200

            st.divider()

            combinatorial_strategy = st.checkbox(
                "🔀 Stratégie combinatoire pour mismatches",
                value=False,
                help="Tente de combiner plusieurs lignes de la base 2 pour matcher les mismatches de la base 1"
            )

            if combinatorial_strategy:
                max_combinations = st.slider(
                    "Nombre max de combinaisons",
                    min_value=2,
                    max_value=5,
                    value=4,
                    help="Nombre maximum de lignes à combiner (2, 3, 4...)"
                )
            else:
                max_combinations = 4

    # Corps principal
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Fichier 1")

        # Choix du mode de sélection
        file1_mode = st.radio(
            "Mode de sélection",
            options=["upload", "path"],
            format_func=lambda x: "📤 Upload/Parcourir" if x == "upload" else "⌨️ Saisir le chemin",
            key="file1_mode",
            horizontal=True
        )

        if file1_mode == "upload":
            uploaded_file1 = st.file_uploader(
                "Choisir un fichier Excel",
                type=["xlsx", "xls"],
                key="file1_uploader",
                help="Glissez-déposez ou cliquez pour parcourir"
            )

            if uploaded_file1:
                # Sauvegarder temporairement le fichier
                temp_path1 = OUTPUT_DIR / uploaded_file1.name
                with open(temp_path1, "wb") as f:
                    f.write(uploaded_file1.getbuffer())
                file1_path = str(temp_path1)
                st.success(f"✅ Fichier chargé : {uploaded_file1.name}")
            else:
                file1_path = ""
        else:
            file1_path = st.text_input(
                "Chemin du fichier",
                value="",
                placeholder="L:\\Test\\Classeur1.xlsx",
                key="file1_path"
            )

        col1a, col1b = st.columns(2)
        with col1a:
            sheet1 = st.text_input("Nom de la feuille", value="Feuil1", key="sheet1")
        with col1b:
            col1 = st.number_input("Numéro de colonne", min_value=1, value=1, key="col1_num")

    with col2:
        st.subheader("📁 Fichier 2")

        # Choix du mode de sélection
        file2_mode = st.radio(
            "Mode de sélection",
            options=["upload", "path"],
            format_func=lambda x: "📤 Upload/Parcourir" if x == "upload" else "⌨️ Saisir le chemin",
            key="file2_mode",
            horizontal=True
        )

        if file2_mode == "upload":
            uploaded_file2 = st.file_uploader(
                "Choisir un fichier Excel",
                type=["xlsx", "xls"],
                key="file2_uploader",
                help="Glissez-déposez ou cliquez pour parcourir"
            )

            if uploaded_file2:
                # Sauvegarder temporairement le fichier
                temp_path2 = OUTPUT_DIR / uploaded_file2.name
                with open(temp_path2, "wb") as f:
                    f.write(uploaded_file2.getbuffer())
                file2_path = str(temp_path2)
                st.success(f"✅ Fichier chargé : {uploaded_file2.name}")
            else:
                file2_path = ""
        else:
            file2_path = st.text_input(
                "Chemin du fichier",
                value="",
                placeholder="L:\\Test\\Classeur2.xlsx",
                key="file2_path"
            )

        col2a, col2b = st.columns(2)
        with col2a:
            sheet2 = st.text_input("Nom de la feuille", value="Feuil1", key="sheet2")
        with col2b:
            col2 = st.number_input("Numéro de colonne", min_value=1, value=1, key="col2_num")

    st.divider()

    # Bouton de lancement
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        run_button = st.button(
            "▶️ Lancer la comparaison",
            type="primary",
            width='stretch'
        )

    # Traitement
    if run_button:
        if not file1_path or not file2_path:
            st.error("⚠️ Veuillez renseigner les deux fichiers à comparer.")
            st.stop()

        try:
            with st.spinner("🔄 Traitement en cours..."):
                # Création des clients
                if offline_mode:
                    st.info(f"📦 Chargement des modèles locaux : {llm_model} + {embedding_model}")
                    emb_client = OfflineEmbeddingsAdapter(embedding_model, logger)

                    llm_client = None
                    # Créer le LLM client si analyse LLM OU stratégie combinatoire activée
                    if llm_equivalent or combinatorial_strategy:
                        llm_client = OfflineLLMAdapter(llm_model, logger)
                        if combinatorial_strategy and not llm_equivalent:
                            st.info("🤖 LLM activé automatiquement pour l'analyse des matches combinatoires")
                else:
                    st.info("🌐 Connexion aux API distantes...")
                    _http_client = httpx.Client(verify=VERIFY_SSL, timeout=httpx.Timeout(300.0))

                    emb_client = DirectOpenAIEmbeddings(
                        model="snowflake-arctic-embed-l-v2.0",
                        api_key=SNOWFLAKE_API_KEY,
                        base_url=SNOWFLAKE_API_BASE,
                        http_client=_http_client,
                        role_prefix=False,
                        logger=logger,
                    )

                    llm_client = None
                    # Créer le LLM client si analyse LLM OU stratégie combinatoire activée
                    if llm_equivalent or combinatorial_strategy:
                        llm_client = DirectOpenAILLM(
                            model="dallem-val",
                            api_key=DALLEM_API_KEY,
                            base_url=DALLEM_API_BASE,
                            http_client=_http_client,
                            logger=logger,
                        )
                        if combinatorial_strategy and not llm_equivalent:
                            st.info("🤖 LLM activé automatiquement pour l'analyse des matches combinatoires")

                # Lecture des fichiers
                progress_bar = st.progress(0, text="📖 Lecture des fichiers Excel...")
                s1_raw = read_excel_col(file1_path, sheet1, col1, logger)
                s2_raw = read_excel_col(file2_path, sheet2, col2, logger)

                if limit and limit > 0:
                    s1_raw = s1_raw[:limit]
                    s2_raw = s2_raw[:limit]

                if not s1_raw or not s2_raw:
                    st.error("❌ Colonnes vides après nettoyage.")
                    st.stop()

                st.success(f"✅ Fichiers chargés : {len(s1_raw)} lignes (fichier 1), {len(s2_raw)} lignes (fichier 2)")

                # Génération des embeddings
                progress_bar.progress(20, text="🔢 Génération des embeddings...")
                D = embed_in_batches(
                    s2_raw,
                    role="doc",
                    batch_size=batch_size,
                    emb_client=emb_client,
                    log=logger,
                    dry_run=False,
                )

                progress_bar.progress(50, text="🔢 Génération des embeddings (requêtes)...")
                Q = embed_in_batches(
                    s1_raw,
                    role="query",
                    batch_size=batch_size,
                    emb_client=emb_client,
                    log=logger,
                    dry_run=False,
                )

                # Matching
                progress_bar.progress(70, text="🔍 Calcul des similarités...")
                if match_mode == 'approx':
                    pairs_topk = cosine_topk_pairs(Q, D, k=topk, log=logger)
                    best_idx, best_val = cosine_two_phase_global_from_pairs(
                        Q.shape[0],
                        D.shape[0],
                        pairs_topk,
                        threshold=threshold,
                        log=logger,
                    )
                else:
                    best_idx, best_val = cosine_two_phase_global(
                        Q,
                        D,
                        threshold=threshold,
                        log=logger,
                    )

                # Construction des résultats
                progress_bar.progress(85, text="📊 Construction des résultats...")
                matches_above, under = [], []
                for i, (j, score) in enumerate(zip(best_idx.tolist(), best_val.tolist())):
                    src = s1_raw[i]
                    if j is not None and j >= 0 and j < len(s2_raw):
                        tgt = s2_raw[j]
                        tgt_idx = j
                    else:
                        tgt = ""
                        tgt_idx = None

                    row = {
                        "src_index": i,
                        "tgt_index": tgt_idx,
                        "source": src,
                        "target": tgt,
                        "score": round(float(score), 4),
                        "match_type": "normal",  # Pour différencier des matches combinatoires
                    }

                    if tgt_idx is not None and score >= threshold:
                        matches_above.append(row)
                    else:
                        under.append(row)

                # Analyse LLM
                if llm_client:
                    progress_bar.progress(90, text="🤖 Analyse LLM en cours...")
                    budget = max(0, int(llm_max))
                    used_above = 0

                    # Analyse des matches
                    if matches_above and budget > 0:
                        n_above = min(len(matches_above), budget)
                        used_above = n_above
                        for k_idx in range(n_above):
                            row = matches_above[k_idx]
                            antago, expl = llm_client.analyse_equivalence(row["source"], row["target"])
                            row["équivalence"] = antago
                            row["commentaire"] = expl
                            row["analyse_llm"] = "Oui (normal)"
                            row.setdefault("promu_par_llm", False)

                        # Pour les matches non analysés
                        for k_idx in range(n_above, len(matches_above)):
                            row = matches_above[k_idx]
                            row.setdefault("équivalence", None)
                            row.setdefault("commentaire", "Non analysé (limite budget LLM)")
                            row.setdefault("analyse_llm", "Non")
                            row.setdefault("promu_par_llm", False)

                    # Analyse des mismatches
                    remaining = max(0, budget - used_above)
                    if under and remaining > 0:
                        n_under = min(len(under), remaining)
                        for idx in range(n_under):
                            row = under[idx]
                            antago, expl = llm_client.analyse_equivalence(row["source"], row["target"])
                            row["équivalence"] = antago
                            row["commentaire"] = expl
                            row["analyse_llm"] = "Oui (mismatch)"
                            row.setdefault("promu_par_llm", False)

                        # Pour les mismatches non analysés
                        for idx in range(n_under, len(under)):
                            row = under[idx]
                            row.setdefault("équivalence", None)
                            row.setdefault("commentaire", "Non analysé (limite budget LLM)")
                            row.setdefault("analyse_llm", "Non")

                        # Promotion
                        used_targets = {r.get("tgt_index") for r in matches_above if r.get("tgt_index") is not None}
                        new_under = []
                        promoted_count = 0

                        for row in under:
                            if row.get("équivalence") and row.get("tgt_index") is not None:
                                tgt_idx = row["tgt_index"]
                                if tgt_idx not in used_targets:
                                    row["promu_par_llm"] = True
                                    matches_above.append(row)
                                    used_targets.add(tgt_idx)
                                    promoted_count += 1
                                else:
                                    new_under.append(row)
                            else:
                                new_under.append(row)

                        under = new_under

                # Stratégie combinatoire pour les mismatches
                if combinatorial_strategy and under:
                    progress_bar.progress(92, text="🔀 Application de la stratégie combinatoire...")
                    st.info(f"🔀 Traitement de {len(under)} mismatches avec stratégie combinatoire...")

                    combinatorial_matches, definitive_mismatches = apply_combinatorial_strategy(
                        mismatches=under,
                        s1_raw=s1_raw,
                        s2_raw=s2_raw,
                        Q=Q,
                        D=D,
                        threshold=threshold,
                        max_combinations=max_combinations,
                        logger=logger,
                    )

                    if combinatorial_matches:
                        st.success(
                            f"✅ {len(combinatorial_matches)} nouveaux matches trouvés par stratégie combinatoire !"
                        )
                        # Ajouter les matches combinatoires aux matches normaux
                        matches_above.extend(combinatorial_matches)

                    # Remplacer under par les mismatches définitifs
                    under = definitive_mismatches

                    # Analyse LLM des matches combinatoires
                    if llm_client and combinatorial_matches:
                        progress_bar.progress(93, text="🤖 Analyse LLM des matches combinatoires...")
                        st.info(f"🤖 Analyse LLM de {len(combinatorial_matches)} matches combinatoires...")

                        for combo_idx, combo_match in enumerate(combinatorial_matches):
                            logger.info(
                                f"[llm-combo] Analyse {combo_idx + 1}/{len(combinatorial_matches)} : "
                                f"src_idx={combo_match['src_index']}"
                            )

                            # Analyser l'équivalence avec le LLM
                            antago, expl = llm_client.analyse_equivalence(
                                combo_match["source"],
                                combo_match["target"]
                            )

                            # Ajouter les résultats LLM au match combinatoire
                            # Utiliser les mêmes noms de colonnes que pour les matches normaux
                            combo_match["équivalence"] = antago
                            combo_match["commentaire"] = expl
                            combo_match["analyse_llm"] = "Oui (combinatoire)"

                            logger.debug(
                                f"[llm-combo] src={combo_match['src_index']}, "
                                f"équivalence={antago}, commentaire={expl[:50]}..."
                            )

                        st.success(
                            f"✅ Analyse LLM terminée pour {len(combinatorial_matches)} matches combinatoires"
                        )

                        # Compter les matches combinatoires validés/rejetés par le LLM
                        validated = sum(1 for m in combinatorial_matches if m.get("équivalence") is True)
                        rejected = sum(1 for m in combinatorial_matches if m.get("équivalence") is False)
                        uncertain = len(combinatorial_matches) - validated - rejected

                        st.info(
                            f"📊 Résultats LLM : {validated} validés ✅, "
                            f"{rejected} rejetés ❌, {uncertain} incertains ⚠️"
                        )

                # Export des résultats
                progress_bar.progress(95, text="💾 Sauvegarde des résultats...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                matches_filename = f"matches_{timestamp}.xlsx"
                under_filename = f"under_{timestamp}.xlsx"

                matches_path = OUTPUT_DIR / matches_filename
                under_path = OUTPUT_DIR / under_filename

                pd.DataFrame(matches_above).to_excel(matches_path, index=False, engine="xlsxwriter")
                pd.DataFrame(under).to_excel(under_path, index=False, engine="xlsxwriter")

                progress_bar.progress(100, text="✅ Terminé !")

                # Affichage des résultats
                st.markdown("---")
                st.markdown("## 🎯 Résultats")

                # Métriques
                total = len(matches_above) + len(under)
                match_rate = (len(matches_above) / total * 100) if total > 0 else 0

                # Compter les matches combinatoires
                combinatorial_count = sum(1 for m in matches_above if m.get("match_type") == "combinatorial")
                normal_matches = len(matches_above) - combinatorial_count

                if combinatorial_count > 0:
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                else:
                    col_m1, col_m2, col_m3 = st.columns(3)

                with col_m1:
                    st.metric(
                        label="✅ Matches normaux",
                        value=normal_matches,
                        help="Paires simples au-dessus du seuil"
                    )

                with col_m2:
                    if combinatorial_count > 0:
                        st.metric(
                            label="🔀 Matches combinatoires",
                            value=combinatorial_count,
                            help="Matches trouvés par combinaison de lignes"
                        )
                    else:
                        st.metric(
                            label="⚠️ Sous le seuil",
                            value=len(under),
                            help="Paires sous le seuil"
                        )

                with col_m3:
                    if combinatorial_count > 0:
                        st.metric(
                            label="⚠️ Mismatches définitifs",
                            value=len(under),
                            help="Aucune combinaison trouvée"
                        )
                    else:
                        st.metric(
                            label="📊 Taux de match",
                            value=f"{match_rate:.1f}%",
                            help="Pourcentage de correspondances"
                        )

                if combinatorial_count > 0:
                    with col_m4:
                        st.metric(
                            label="📊 Taux de match",
                            value=f"{match_rate:.1f}%",
                            help="Pourcentage de correspondances"
                        )

                st.markdown("---")

                # Aperçu des résultats
                tab1, tab2 = st.tabs(["✅ Matches", "⚠️ Mismatches définitifs"])

                with tab1:
                    st.subheader(f"Correspondances (≥ {threshold})")
                    if matches_above:
                        # Séparer matches normaux et combinatoires
                        normal_matches_list = [m for m in matches_above if m.get("match_type") != "combinatorial"]
                        combinatorial_matches_list = [m for m in matches_above if m.get("match_type") == "combinatorial"]

                        if combinatorial_count > 0:
                            st.info(
                                f"📊 Total : {len(matches_above)} matches "
                                f"({normal_matches} normaux + {combinatorial_count} combinatoires)"
                            )

                        # Afficher les matches combinatoires en premier avec un avertissement
                        if combinatorial_matches_list:
                            st.warning(
                                f"⚠️ {len(combinatorial_matches_list)} match(es) combinatoire(s) détecté(s) - "
                                "Une ligne de base 1 correspond à plusieurs lignes combinées de base 2"
                            )
                            st.markdown("**🔀 Matches combinatoires :**")
                            df_combo = pd.DataFrame(combinatorial_matches_list)
                            st.dataframe(
                                df_combo,
                                width='stretch',
                                height=min(200, len(combinatorial_matches_list) * 50 + 50)
                            )

                        # Afficher les matches normaux
                        if normal_matches_list:
                            if combinatorial_matches_list:
                                st.markdown("**✅ Matches normaux :**")
                            df_normal = pd.DataFrame(normal_matches_list)
                            st.dataframe(
                                df_normal,
                                width='stretch',
                                height=min(400, len(normal_matches_list) * 50 + 50)
                            )

                        # Si aucun match combinatoire, afficher tout ensemble
                        if not combinatorial_matches_list:
                            df_matches = pd.DataFrame(matches_above)
                            st.dataframe(df_matches, width='stretch', height=400)
                    else:
                        st.info("Aucune correspondance au-dessus du seuil.")

                with tab2:
                    if combinatorial_strategy:
                        st.subheader(f"Mismatches définitifs")
                        st.info("Ces lignes n'ont pas trouvé de correspondance, même avec la stratégie combinatoire")
                    else:
                        st.subheader(f"Sous le seuil (< {threshold})")

                    if under:
                        df_under = pd.DataFrame(under)
                        st.dataframe(df_under, width='stretch', height=400)
                    else:
                        st.success("✅ Toutes les lignes ont trouvé une correspondance !")

                # Téléchargements
                st.markdown("---")
                st.subheader("📥 Télécharger les résultats")

                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    with open(matches_path, "rb") as f:
                        st.download_button(
                            label="📥 Télécharger matches.xlsx",
                            data=f,
                            file_name=matches_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )

                with col_d2:
                    with open(under_path, "rb") as f:
                        st.download_button(
                            label="📥 Télécharger under_threshold.xlsx",
                            data=f,
                            file_name=under_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                st.success(f"✅ Résultats sauvegardés dans : {OUTPUT_DIR}")

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            with st.expander("🔍 Détails de l'erreur"):
                st.code(traceback.format_exc())
            logger.error(f"Erreur : {e}")
            logger.debug(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6b7280; padding: 20px;'>
            <p>CompareDB v2.0 - Comparaison sémantique avec IA</p>
            <p>Mode : {} | Modèles : {}</p>
        </div>
        """.format(
            "🔌 Hors ligne" if offline_mode else "🌐 En ligne",
            f"{llm_model} + {embedding_model}" if offline_mode else "API distantes"
        ),
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
