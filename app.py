"""
Serveur Flask pour l'interface web de CompareDB.
Gère les modes en ligne et hors ligne pour la comparaison sémantique.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import sys
import logging
import traceback
import uuid
from typing import Optional, List
import numpy as np
import pandas as pd
import httpx

# Import du code existant
from test2_v4 import (
    read_excel_col,
    embed_in_batches,
    cosine_two_phase_global,
    cosine_topk_pairs,
    cosine_two_phase_global_from_pairs,
    DirectOpenAIEmbeddings,
    DirectOpenAILLM,
    normalize_text,
    make_logger,
)

# Import du mode offline
try:
    from offline_models import OfflineEmbeddingsAdapter, OfflineLLMAdapter
    OFFLINE_AVAILABLE = True
except ImportError as e:
    OFFLINE_AVAILABLE = False
    print(f"Mode offline non disponible: {e}")

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logger
logger = make_logger(debug=True)


@app.route('/')
def index():
    """Page d'accueil."""
    return send_from_directory('static', 'index.html')


@app.route('/api/status')
def status():
    """Statut de l'application."""
    return jsonify({
        'status': 'ok',
        'offline_available': OFFLINE_AVAILABLE,
        'version': '2.0'
    })


@app.route('/api/compare', methods=['POST'])
def compare():
    """
    Endpoint principal de comparaison.
    Accepte un JSON avec les paramètres de comparaison.
    """
    try:
        data = request.json
        logger.info(f"[api] Nouvelle requête de comparaison - mode: {data.get('mode', 'online')}")

        # Validation
        required_fields = ['file1', 'sheet1', 'col1', 'file2', 'sheet2', 'col2']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400

        # Paramètres
        mode = data.get('mode', 'online')
        file1 = data['file1']
        sheet1 = data['sheet1']
        col1 = int(data['col1'])
        file2 = data['file2']
        sheet2 = data['sheet2']
        col2 = int(data['col2'])
        threshold = float(data.get('threshold', 0.78))
        batch_size = int(data.get('batch_size', 16))
        limit = data.get('limit')
        llm_equivalent = data.get('llm_equivalent', False)
        match_mode = data.get('match_mode', 'full')
        topk = int(data.get('topk', 10))

        # Validation du mode offline
        if mode == 'offline' and not OFFLINE_AVAILABLE:
            return jsonify({
                'error': 'Mode hors ligne non disponible. Installez les dépendances requises.'
            }), 400

        # Création des clients
        if mode == 'offline':
            logger.info("[api] Mode hors ligne activé")
            emb_model_key = data.get('embedding_model', 'bge-m3')
            llm_model_key = data.get('llm_model', 'qwen')

            emb_client = OfflineEmbeddingsAdapter(emb_model_key, logger)

            llm_client = None
            if llm_equivalent:
                llm_client = OfflineLLMAdapter(llm_model_key, logger)

        else:
            logger.info("[api] Mode en ligne activé")
            # Configuration API (depuis variables d'environnement ou hardcodées)
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
            if llm_equivalent:
                llm_client = DirectOpenAILLM(
                    model="dallem-val",
                    api_key=DALLEM_API_KEY,
                    base_url=DALLEM_API_BASE,
                    http_client=_http_client,
                    logger=logger,
                )

        # Lecture des fichiers Excel
        logger.info(f"[api] Lecture des fichiers Excel")
        s1_raw = read_excel_col(file1, sheet1, col1, logger)
        s2_raw = read_excel_col(file2, sheet2, col2, logger)

        if limit:
            limit = int(limit)
            s1_raw = s1_raw[:limit]
            s2_raw = s2_raw[:limit]

        if not s1_raw or not s2_raw:
            return jsonify({'error': 'Colonnes vides après nettoyage'}), 400

        logger.info(f"[api] Fichiers chargés - Base1={len(s1_raw)}, Base2={len(s2_raw)}")

        # Génération des embeddings
        logger.info(f"[api] Génération des embeddings")
        D = embed_in_batches(
            s2_raw,
            role="doc",
            batch_size=batch_size,
            emb_client=emb_client,
            log=logger,
            dry_run=False,
        )
        Q = embed_in_batches(
            s1_raw,
            role="query",
            batch_size=batch_size,
            emb_client=emb_client,
            log=logger,
            dry_run=False,
        )

        # Matching
        logger.info(f"[api] Matching - mode={match_mode}")
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
            }

            if tgt_idx is not None and score >= threshold:
                matches_above.append(row)
            else:
                under.append(row)

        # Analyse LLM (optionnelle)
        if llm_client:
            logger.info(f"[api] Analyse LLM - matches={len(matches_above)}, under={len(under)}")
            llm_max = data.get('llm_max', 200)
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
                    row.setdefault("promu_par_llm", False)

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
                logger.info(f"[api] Promotion de {promoted_count} mismatches")

        # Export des résultats
        session_id = str(uuid.uuid4())[:8]
        matches_filename = f"matches_{session_id}.xlsx"
        under_filename = f"under_{session_id}.xlsx"

        matches_path = os.path.join(OUTPUT_DIR, matches_filename)
        under_path = os.path.join(OUTPUT_DIR, under_filename)

        pd.DataFrame(matches_above).to_excel(matches_path, index=False, engine="xlsxwriter")
        pd.DataFrame(under).to_excel(under_path, index=False, engine="xlsxwriter")

        logger.info(f"[api] Comparaison terminée - matches={len(matches_above)}, under={len(under)}")

        return jsonify({
            'success': True,
            'matches_count': len(matches_above),
            'under_count': len(under),
            'matches_file': f'/api/download/{matches_filename}',
            'under_file': f'/api/download/{under_filename}',
        })

    except Exception as e:
        logger.error(f"[api] Erreur: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download(filename):
    """Téléchargement des fichiers de résultats."""
    try:
        return send_file(
            os.path.join(OUTPUT_DIR, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"[api] Erreur de téléchargement: {e}")
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    print("=" * 60)
    print("CompareDB - Interface Web")
    print("=" * 60)
    print(f"Mode offline disponible: {OFFLINE_AVAILABLE}")
    print(f"Répertoire de sortie: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nDémarrage du serveur...")
    print("Interface disponible sur: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
