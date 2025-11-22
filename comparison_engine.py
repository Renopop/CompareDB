import os
import re
import argparse
import sys
import traceback
import math
import time
from typing import List, Optional, Tuple, Pattern

import numpy as np
import pandas as pd
import httpx
from numpy.linalg import norm
from openai import OpenAI
import openai
import logging
from logging import Logger
from collections import Counter


# =========================
#  Config (forcées + env)
# =========================
LLM_MODEL = "dallem-val"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 16  # taille batch embeddings (modifiable par --batch-size)

# Active ou non la détection des equivalences par défaut (sans CLI)
LLM_ANTAGONISM_ENABLED = True  # passe à True pour activer par défaut

HARDCODE = {
    "DALLEM_API_BASE": "https://api.dev.dassault-aviation.pro/dallem-pilote/v1",
    "SNOWFLAKE_API_BASE": "https://api.dev.dassault-aviation.pro/snowflake-arctic-embed-l-v2.0/v1",
    "DALLEM_API_KEY": "EMPTY",    # à surcharger par l'env
    "SNOWFLAKE_API_KEY": "token",  # à surcharger par l'env
    "DISABLE_SSL_VERIFY": "true",
    "RERANK_API_BASE": "https://api.dev.dassault-aviation.pro/bge-reranker-v2-m3/v1/",
    "RERANK_API_KEY": "EMPTY",
}

DALLEM_API_BASE = os.getenv("DALLEM_API_BASE", HARDCODE["DALLEM_API_BASE"]).rstrip("/")
SNOWFLAKE_API_BASE = os.getenv("SNOWFLAKE_API_BASE", HARDCODE["SNOWFLAKE_API_BASE"]).rstrip("/")
DALLEM_API_KEY = os.getenv("DALLEM_API_KEY", HARDCODE["DALLEM_API_KEY"])
SNOWFLAKE_API_KEY = os.getenv("SNOWFLAKE_API_KEY", HARDCODE["SNOWFLAKE_API_KEY"])
RERANK_API_BASE = os.getenv("RERANK_API_BASE", HARDCODE["RERANK_API_BASE"]).rstrip("/")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", HARDCODE["RERANK_API_KEY"])

VERIFY_SSL = not (
    os.getenv("DISABLE_SSL_VERIFY", HARDCODE["DISABLE_SSL_VERIFY"])
    .lower()
    in ("1", "true", "yes", "on")
)


# =========================
#  Logging espion
# =========================
def _mask(s: Optional[str]) -> str:
    if not s:
        return "<vide>"
    if len(s) <= 6:
        return "***"
    return s[:3] + "…" + s[-3:]


def make_logger(debug: bool) -> Logger:
    log = logging.getLogger("compareur")
    log.setLevel(logging.DEBUG if debug else logging.INFO)

    if log.handlers:
        return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler("compareur_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)
    return log


# =========================
#  Client embeddings direct
# =========================
class DirectOpenAIEmbeddings:
    """
    Client embeddings minimal (OpenAI v1-compatible).
    role_prefix=True -> "passage:" / "query:".
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        http_client: Optional[httpx.Client] = None,
        role_prefix: bool = True,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.role_prefix = role_prefix
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.log = logger or logging.getLogger("compareur")

    def _apply_prefix(self, items: List[str], role: str) -> List[str]:
        if not self.role_prefix:
            return items
        pref = "query: " if role == "query" else "passage: "
        return [pref + (x or "") for x in items]

    def _retry_request(self, func, max_retries: int = 5, base_delay: float = 1.0):
        """
        Exécute func() avec retry exponentiel sur les erreurs transitoires.
        - Retry sur : APIConnectionError, RateLimitError, APIError
        - Pas de retry sur : AuthenticationError, NotFoundError, etc.
        """
        for attempt in range(max_retries):
            try:
                return func()
            except (openai.APIConnectionError, openai.RateLimitError, openai.APIError) as e:
                if attempt == max_retries - 1:
                    self.log.error(
                        f"[embeddings] Échec après {max_retries} tentatives — {type(e).__name__}: {e}"
                    )
                    raise
                wait_time = base_delay * (2 ** attempt)
                self.log.warning(
                    f"[embeddings] Tentative {attempt + 1}/{max_retries} échouée "
                    f"({type(e).__name__}: {e}) — retry dans {wait_time:.1f}s"
                )
                time.sleep(wait_time)

    def _create_embeddings(self, inputs: List[str]) -> List[List[float]]:
        t0 = time.time()
        self.log.debug(
            f"[embeddings] POST {self.client.base_url} | model={self.model} "
            f"| n_inputs={len(inputs)} | len0={len(inputs[0]) if inputs else 0}"
        )

        def _do_request():
            return self.client.embeddings.create(model=self.model, input=inputs)

        try:
            resp = self._retry_request(_do_request)
            dur = (time.time() - t0) * 1000
            self.log.debug(
                f"[embeddings] OK in {dur:.1f} ms | items={len(resp.data)} "
                f"| dim≈{len(resp.data[0].embedding) if resp.data else 'n/a'}"
            )
            return [d.embedding for d in resp.data]

        except openai.NotFoundError as e:
            self.log.error(f"[embeddings] NotFoundError (modèle='{self.model}' ?) : {e}")
            self.log.debug(traceback.format_exc())
            raise
        except openai.AuthenticationError as e:
            self.log.error("[embeddings] AuthenticationError — clé invalide ?")
            self.log.debug(traceback.format_exc())
            raise
        except Exception as e:
            self.log.error(f"[embeddings] Exception — {e}")
            self.log.debug(traceback.format_exc())
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="passage")
        return self._create_embeddings(inputs)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="query")
        return self._create_embeddings(inputs)


# =========================
#  Client LLM (DALLEM)
# =========================
class DirectOpenAILLM:
    """
    Client LLM minimal pour DALLEM (OpenAI chat completions compatible).
    Utilisé pour détecter si deux exigences sont equivalent.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        http_client: Optional[httpx.Client] = None,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http = http_client
        self.log = logger or logging.getLogger("compareur")

    def analyse_equivalence(self, text1: str, text2: str) -> Tuple[Optional[bool], str]:
        url = self.base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_msg = (
            "Tu es un expert en exigences aéronautiques. "
            "On te donne deux exigences techniques. "
            "Tu dois répondre si elles expriment la même chose (ou quasi) en 15 mots. "
        )

        user_msg = (
            "Exigence A :\n"
            f"{text1}\n\n"
            "Exigence B :\n"
            f"{text2}\n\n"
            "Les deux exigences expriment-elles la même chose ou quasiment, ta reponse doit impérativement commencer par TRUE ou FALSE ? "
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }

        try:
            resp = self.http.post(url, headers=headers, json=payload, timeout=120.0)
            resp.raise_for_status()
        except Exception as e:
            self.log.error(f"[llm] Erreur HTTP DALLEM: {e}")
            self.log.debug(traceback.format_exc())
            return None, f"ERREUR HTTP LLM: {e}"

        try:
            data = resp.json()
        except Exception:
            self.log.error("[llm] Impossible de parser la réponse JSON DALLEM", exc_info=True)
            return None, "ERREUR LLM: réponse vide ou non JSON"

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except Exception:
            self.log.error("[llm] Format de réponse DALLEM inattendu", exc_info=True)
            return None, "ERREUR LLM: format inattendu"

        self.log.debug(f"[llm] Réponse DALLEM brute: {content!r}")

        tokens = content.split()
        first = tokens[0].lower() if tokens else ""
        antagoniste: Optional[bool]
        if "true" in first:
            antagoniste = True
        elif "false" in first:
            antagoniste = False
        else:
            antagoniste = None

        return antagoniste, content


# =========================
#  Utils texte & Excel
# =========================
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def read_excel_col(path: str, sheet: str, col_1based: int, log: Logger) -> List[str]:
    try:
        log.info(
            f"[excel] lecture: file='{path}' | sheet='{sheet}' | col={col_1based}"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier inexistant: {path}")
        df = pd.read_excel(path, sheet_name=sheet, dtype=object, header=None)
        log.debug(
            f"[excel] colonnes détectées ({len(df.columns)}): {list(df.columns)}"
        )
        if col_1based < 1 or col_1based > len(df.columns):
            raise ValueError(
                f"Colonne {col_1based} invalide pour {os.path.basename(path)} "
                f"(1..{len(df.columns)})"
            )
        col = df.iloc[:, col_1based - 1]
        values = [
            normalize_text(str(x))
            for x in col.dropna().astype(str).tolist()
            if normalize_text(str(x))
        ]
        sample = values[:3]
        log.debug(f"[excel] lignes utiles={len(values)} | sample={sample}")
        return values
    except Exception as e:
        log.error(f"[excel] Erreur: {e}")
        log.debug(traceback.format_exc())
        raise


def embed_in_batches(
    texts: List[str],
    role: str,
    batch_size: int,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool = False,
) -> np.ndarray:
    out: List[List[float]] = []
    n = len(texts)
    log.info(
        f"[emb] start role={role} | n={n} | batch_size={batch_size} | dry_run={dry_run}"
    )
    for i in range(0, n, batch_size):
        chunk = texts[i: i + batch_size]
        log.debug(
            f"[emb] chunk {i // batch_size + 1}/{math.ceil(n / max(1, batch_size))} "
            f"| size={len(chunk)} | first='{(chunk[0][:120] if chunk else '')}'"
        )
        try:
            if dry_run:
                dim = 1024
                fake = np.random.rand(len(chunk), dim).astype(np.float32) - 0.5
                out.extend(fake.tolist())
            else:
                if role == "query":
                    out.extend(emb_client.embed_queries(chunk))
                else:
                    out.extend(emb_client.embed_documents(chunk))
        except Exception as e:
            log.error(f"[emb] échec sur le chunk (i={i}) — {e}")
            log.debug(traceback.format_exc())
            raise

    M = np.asarray(out, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] != n:
        log.error(f"[emb] shape inattendue: {M.shape} (attendu ({n}, d))")

    denom = norm(M, axis=1, keepdims=True) + 1e-12
    if np.any(np.isnan(denom)):
        log.warning("[emb] NaN détecté dans la norme, correction appliquée.")
        denom = np.nan_to_num(denom, nan=1.0)
    M = M / denom
    log.info(
        f"[emb] terminé | shape={M.shape} | d={M.shape[1] if M.ndim == 2 else 'n/a'}"
    )
    return M


# =========================
#  Matching global 2 phases (matches + missmatchs) avec unicité
# =========================
def cosine_two_phase_global(
    Q: np.ndarray,
    D: np.ndarray,
    threshold: float,
    log: Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matching global en deux phases avec unicité.

    Étape 0 : on calcule la matrice complète des cosinus : sims[i, j].

    Phase 1 (matches) :
      - on construit la liste de tous les couples (i, j) tels que sims[i, j] >= threshold
      - on trie cette liste par score décroissant
      - on parcourt dans l'ordre et on prend un couple (i, j) seulement si
        i et j ne sont pas encore utilisés
      -> ces couples sont les "matches" (>= seuil), unicité garantie.

    Phase 2 (missmatchs) :
      - on regarde toutes les sources i non utilisées en phase 1
      - toutes les cibles j non utilisées en phase 1
      - on construit la liste de tous les couples (i, j) possibles restants
      - on trie cette liste par score décroissant
      - on parcourt dans l'ordre et on prend un couple (i, j) seulement si
        i et j ne sont pas encore utilisés (au sein de la phase 2)
      -> on obtient des paires supplémentaires avec unicité globale.

    Au final :
      - chaque i a au plus une j
      - chaque j a au plus un i
      - les matches forts (>= seuil) sont choisis globalement en priorité.
    """

    log.info(f"[cosine-2phase-global] calc sims | Q={Q.shape} | D={D.shape} | threshold={threshold}")
    sims = np.matmul(Q, D.T)
    if np.any(np.isnan(sims)):
        log.warning("[cosine-2phase-global] NaN dans la matrice — correction NaN->0")
        sims = np.nan_to_num(sims, nan=0.0)

    n_q, n_d = sims.shape

    # Résultats initiaux
    best_idx = np.full(n_q, -1, dtype=np.int64)
    best_val = np.zeros(n_q, dtype=np.float32)

    used_src = np.zeros(n_q, dtype=bool)
    used_tgt = np.zeros(n_d, dtype=bool)

    # -----------------------
    # Phase 1 : matches (>= seuil)
    # -----------------------
    pairs_match: List[Tuple[float, int, int]] = []
    for i in range(n_q):
        for j in range(n_d):
            val = float(sims[i, j])
            if val >= threshold:
                pairs_match.append((val, i, j))

    pairs_match.sort(key=lambda x: x[0], reverse=True)

    for val, i, j in pairs_match:
        if used_src[i] or used_tgt[j]:
            continue
        best_idx[i] = j
        best_val[i] = val
        used_src[i] = True
        used_tgt[j] = True

    # -----------------------
    # Phase 2 : missmatchs (restant, sans seuil)
    # -----------------------
    remaining_src = np.where(~used_src)[0]
    remaining_tgt = np.where(~used_tgt)[0]

    if len(remaining_src) > 0 and len(remaining_tgt) > 0:
        pairs_miss: List[Tuple[float, int, int]] = []
        for i in remaining_src:
            for j in remaining_tgt:
                val = float(sims[i, j])
                pairs_miss.append((val, i, j))

        pairs_miss.sort(key=lambda x: x[0], reverse=True)

        used_src2 = set()
        used_tgt2 = set()
        for val, i, j in pairs_miss:
            if i in used_src2 or j in used_tgt2:
                continue
            best_idx[i] = j
            best_val[i] = val
            used_src2.add(i)
            used_tgt2.add(j)

    log.debug(
        f"[cosine-2phase-global] stats: min={float(best_val.min()):.4f} | "
        f"max={float(best_val.max()):.4f} | mean={float(best_val.mean()):.4f}"
    )
    return best_idx, best_val


# =========================
#  Matching approx top-k (optionnel)
# =========================
def cosine_topk_pairs(
    Q: np.ndarray,
    D: np.ndarray,
    k: int,
    log: Logger,
) -> List[Tuple[float, int, int]]:
    """
    Approx : pour chaque source i, ne garder que les top-k cibles.
    Utilise uniquement NumPy (CPU).
    Retourne une liste de tuples (score, i, j).
    """
    n_q, d_q = Q.shape
    n_d, d_d = D.shape
    if d_q != d_d:
        raise ValueError(f"Dim mismatch Q={Q.shape}, D={D.shape}")

    if k <= 0:
        k = 1
    k = min(k, n_d)

    log.info(f"[cosine-topk] start | n_q={n_q} | n_d={n_d} | k={k}")

    pairs: List[Tuple[float, int, int]] = []

    for i in range(n_q):
        v = Q[i]  # (d,)
        sims = D @ v  # (n_d,)
        if np.any(np.isnan(sims)):
            log.warning(f"[cosine-topk] NaN détecté sur la ligne {i}, correction NaN->0")
            sims = np.nan_to_num(sims, nan=0.0)

        if k >= n_d:
            top_idx = np.argsort(sims)[::-1]
        else:
            idx_part = np.argpartition(sims, -k)[-k:]
            top_idx = idx_part[np.argsort(sims[idx_part])[::-1]]

        for j in top_idx:
            pairs.append((float(sims[j]), i, int(j)))

    log.info(f"[cosine-topk] terminé | nb_pairs={len(pairs)}")
    return pairs


def cosine_two_phase_global_from_pairs(
    n_q: int,
    n_d: int,
    pairs: List[Tuple[float, int, int]],
    threshold: float,
    log: Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Variante du matching global 2 phases en partant d'une liste de paires (score, i, j)
    (typiquement top-k approx).

    Retourne best_idx, best_val comme cosine_two_phase_global().
    """
    log.info(
        f"[cosine-2phase-from-pairs] n_q={n_q} | n_d={n_d} | "
        f"nb_pairs={len(pairs)} | threshold={threshold}"
    )

    best_idx = np.full(n_q, -1, dtype=np.int64)
    best_val = np.zeros(n_q, dtype=np.float32)

    used_src = np.zeros(n_q, dtype=bool)
    used_tgt = np.zeros(n_d, dtype=bool)

    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Phase 1 : >= seuil, unicité
    for val, i, j in pairs_sorted:
        if val < threshold:
            continue
        if i < 0 or i >= n_q or j < 0 or j >= n_d:
            continue
        if used_src[i] or used_tgt[j]:
            continue
        best_idx[i] = j
        best_val[i] = val
        used_src[i] = True
        used_tgt[j] = True

    # Phase 2 : compléments sur sources/cibles restantes
    remaining_src = np.where(~used_src)[0]
    remaining_tgt = np.where(~used_tgt)[0]

    if len(remaining_src) > 0 and len(remaining_tgt) > 0:
        used_src2 = set()
        used_tgt2 = set()
        for val, i, j in pairs_sorted:
            if i not in remaining_src or j not in remaining_tgt:
                continue
            if i in used_src2 or j in used_tgt2:
                continue
            best_idx[i] = j
            best_val[i] = val
            used_src2.add(i)
            used_tgt2.add(j)

    log.debug(
        f"[cosine-2phase-from-pairs] stats: min={float(best_val.min()):.4f} | "
        f"max={float(best_val.max()):.4f} | mean={float(best_val.mean()):.4f}"
    )
    return best_idx, best_val


# =========================
#  Corrélation forcée
# =========================
CorrRule = Tuple[Pattern[str], str]


def build_default_corr_rules(log: Logger) -> List[CorrRule]:
    rules: List[CorrRule] = []

    rules.append((
        re.compile(r"\bthe function\s+(SF_[A-Za-z0-9_]+)\b", flags=re.IGNORECASE),
        r"the system \1",
    ))
    rules.append((
        re.compile(r"\bthe function\b", flags=re.IGNORECASE),
        "the system",
    ))
    rules.append((
        re.compile(r"\bfunction\b", flags=re.IGNORECASE),
        "system",
    ))

    log.info(f"[corr] {len(rules)} règle(s) de corrélation par défaut actives")
    return rules


def apply_corr_rules(text: str, rules: List[CorrRule]) -> str:
    out = text
    for pattern, replacement in rules:
        out = pattern.sub(replacement, out)
    return out


def collect_corr_rules_from_user(log: Logger) -> List[CorrRule]:
    print("\n=== Définition manuelle des règles de corrélation ===")
    print("Saisissez des paires 'source;target' (ligne vide pour finir).")
    print("Exemple :")
    print("  the function;the system\n")

    rules: List[CorrRule] = []
    while True:
        try:
            line = input("Règle manuelle (source;target) ou vide pour terminer: ").strip()
        except EOFError:
            break
        if not line:
            break
        if ";" not in line:
            print("  ⚠ Format attendu: source;target (par ex. 'the function;the system')")
            continue
        src, tgt = line.split(";", 1)
        src = src.strip()
        tgt = tgt.strip()
        if not src or not tgt:
            print("  ⚠ Source et target ne doivent pas être vides.")
            continue

        pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
        rules.append((pattern, tgt))
        log.info(f"[corr-manuel] Règle ajoutée: '{src}' -> '{tgt}'")

    if not rules:
        print("Aucune règle manuelle saisie.")
    else:
        print(f"{len(rules)} règle(s) manuelle(s) définie(s).")

    return rules


# =========================
#  Assistant de corrélation (mots + bouts de phrase)
# =========================

STOPWORDS = {"the", "and", "or", "to", "of", "a", "an", "shall", "be", "is", "are"}


def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower())


# --- Mots simples ---
def find_candidate_terms(
    s1: List[str],
    min_freq: int = 3,
    max_terms: int = 20,
    min_len: int = 3,
) -> List[Tuple[str, int]]:
    cnt = Counter()
    for line in s1:
        tokens = tokenize(line)
        for t in tokens:
            t = t.strip()
            if len(t) < min_len:
                continue
            tl = t.lower()
            if tl in STOPWORDS:
                continue
            cnt[tl] += 1

    candidates = [(t, f) for t, f in cnt.items() if f >= min_freq]
    candidates.sort(key=lambda x: -x[1])

    return candidates[:max_terms]


def collect_target_tokens_for_term(
    term: str,
    s1_raw: List[str],
    s2_raw: List[str],
    best_idx: np.ndarray,
) -> Counter:
    term_l = term.lower()
    cnt = Counter()
    for i, src in enumerate(s1_raw):
        if term_l in src.lower():
            j = int(best_idx[i])
            if j < 0 or j >= len(s2_raw):
                continue
            tgt = s2_raw[j]
            tokens = tokenize(tgt)
            for tok in tokens:
                cnt[tok] += 1
    return cnt


def suggest_target_for_term(
    term: str,
    s1_raw: List[str],
    s2_raw: List[str],
    best_idx: np.ndarray,
    min_hits: int = 2,
) -> Tuple[Optional[str], int]:
    cnt = collect_target_tokens_for_term(term, s1_raw, s2_raw, best_idx)
    if not cnt:
        return None, 0

    for w in list(cnt.keys()):
        if w in STOPWORDS or len(w) < 3 or w == term.lower():
            del cnt[w]

    if not cnt:
        return None, 0

    best_word, freq = cnt.most_common(1)[0]
    if freq < min_hits:
        return None, freq

    return best_word, freq


def interactive_edit_suggestions(
    suggestions: List[Tuple[str, str, int]]
) -> List[Tuple[str, str]]:
    print("\n=== Édition des corrélations proposées ===")
    print("Pour chaque ligne :")
    print("  - ENTER : conserver la règle proposée telle quelle")
    print("  - 's'   : supprimer la proposition (pas de règle)")
    print("  - 'new_tgt' : changer uniquement la cible")
    print("  - 'new_src;new_tgt' : changer source ET cible\n")

    final_rules: List[Tuple[str, str]] = []

    for idx, (src, tgt, hits) in enumerate(suggestions, start=1):
        prompt = (
            f"[{idx}] '{src}' -> '{tgt}'"
            + (f" (observé ≈{hits} fois)" if hits is not None else "")
            + "\n   Action (ENTER=ok, 's'=supprimer, 'new_tgt' ou 'new_src;new_tgt'): "
        )
        try:
            ans = input(prompt).strip()
        except EOFError:
            ans = ""

        if ans == "":
            final_rules.append((src, tgt))
        elif ans.lower() == "s":
            continue
        else:
            if ";" in ans:
                new_src, new_tgt = ans.split(";", 1)
                new_src = new_src.strip()
                new_tgt = new_tgt.strip()
                if new_src and new_tgt:
                    final_rules.append((new_src, new_tgt))
            else:
                new_tgt = ans.strip()
                if new_tgt:
                    final_rules.append((src, new_tgt))

    print(f"\n[assist-corr] {len(final_rules)} corrélation(s) retenue(s) après édition.")
    return final_rules


def interactive_correlation_suggestions(
    s1_raw: List[str],
    s2_raw: List[str],
    best_idx: np.ndarray,
    min_term_freq: int = 3,
    min_hits: int = 2,
    max_terms: int = 20,
) -> List[Tuple[str, str]]:
    candidate_terms = find_candidate_terms(
        s1_raw,
        min_freq=min_term_freq,
        max_terms=max_terms,
        min_len=3,
    )
    if not candidate_terms:
        print("\n[assist-corr] Aucun mot candidat détecté pour suggestions automatiques.")
        return []

    print("\n=== Assistant de corrélation automatique (mots simples) ===")
    print("(Suggestions basées sur les matches observés entre le 1er et le 2e document.)")

    raw_suggestions: List[Tuple[str, str, int]] = []

    for term, global_freq in candidate_terms:
        target_guess, hits = suggest_target_for_term(
            term, s1_raw, s2_raw, best_idx, min_hits=min_hits
        )
        if not target_guess:
            continue
        raw_suggestions.append((term, target_guess, hits))

    if not raw_suggestions:
        print("\n[assist-corr] Aucune association suffisamment stable détectée.")
        return []

    print("\nPropositions initiales (avant édition) :")
    for idx, (src, tgt, hits) in enumerate(raw_suggestions, start=1):
        print(f"  [{idx}] '{src}' -> '{tgt}' (observé ≈{hits} fois)")

    final_pairs = interactive_edit_suggestions(raw_suggestions)
    return final_pairs


def build_rules_from_terms(
    rules: List[Tuple[str, str]],
    log: Logger,
) -> List[CorrRule]:
    corr_rules: List[CorrRule] = []
    for src, tgt in rules:
        pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
        corr_rules.append((pattern, tgt))
        log.info(f"[assist-corr] Règle ajoutée: '{src}' -> '{tgt}'")
    return corr_rules


def run_corr_assistant(
    s1_raw: List[str],
    s2_raw: List[str],
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    batch_size: int,
    dry_run: bool,
    min_term_freq: int = 3,
    min_hits: int = 2,
    max_terms: int = 20,
) -> List[CorrRule]:
    if dry_run:
        log.warning("[assist-corr] Ignoré en mode dry-run.")
        return []

    log.info("[assist-corr] Démarrage de l'assistant de corrélation (mots)…")

    D0 = embed_in_batches(
        s2_raw,
        role="doc",
        batch_size=batch_size,
        emb_client=emb_client,
        log=log,
        dry_run=False,
    )
    Q0 = embed_in_batches(
        s1_raw,
        role="query",
        batch_size=batch_size,
        emb_client=emb_client,
        log=log,
        dry_run=False,
    )

    # Correspondance unique globale juste pour analyser les cooccurrences
    best_idx0, _ = cosine_two_phase_global(Q0, D0, threshold=0.0, log=log)

    term_pairs = interactive_correlation_suggestions(
        s1_raw=s1_raw,
        s2_raw=s2_raw,
        best_idx=best_idx0,
        min_term_freq=min_term_freq,
        min_hits=min_hits,
        max_terms=max_terms,
    )

    if not term_pairs:
        return []

    corr_rules = build_rules_from_terms(term_pairs, log)
    return corr_rules


# --- Bouts de phrase (n-grammes corrélés) ---
def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def find_candidate_phrases_correlated(
    s1_raw: List[str],
    best_idx: np.ndarray,
    min_phrase_freq: int = 2,
    max_phrases: int = 20,
    n: int = 5,
) -> List[Tuple[str, int]]:
    phrase_counter = Counter()

    for text in s1_raw:
        tokens = tokenize(text)
        ngrams_i = set(_extract_ngrams(tokens, n))
        for ph in ngrams_i:
            phrase_counter[ph] += 1

    candidates = [(ph, freq) for ph, freq in phrase_counter.items() if freq >= min_phrase_freq]
    candidates.sort(key=lambda x: -x[1])

    return candidates[:max_phrases]


def collect_target_phrases_for_phrase(
    phrase: str,
    s1_raw: List[str],
    s2_raw: List[str],
    best_idx: np.ndarray,
    n: int = 5,
) -> Counter:
    phrase_l = phrase.lower()
    cnt = Counter()

    for i, src in enumerate(s1_raw):
        tokens1 = tokenize(src)
        ngrams1 = _extract_ngrams(tokens1, n)
        if phrase_l not in [ng.lower() for ng in ngrams1]:
            continue

        j = int(best_idx[i])
        if j < 0 or j >= len(s2_raw):
            continue
        tgt = s2_raw[j]
        tokens2 = tokenize(tgt)
        ngrams2 = _extract_ngrams(tokens2, n)

        for ph2 in ngrams2:
            cnt[ph2] += 1

    return cnt


def suggest_target_phrase(
    phrase: str,
    s1_raw: List[str],
    s2_raw: List[str],
    best_idx: np.ndarray,
    n: int = 5,
    min_hits: int = 1,
) -> Tuple[Optional[str], int]:
    cnt = collect_target_phrases_for_phrase(phrase, s1_raw, s2_raw, best_idx, n=n)
    if not cnt:
        return None, 0

    best_phrase, freq = cnt.most_common(1)[0]
    if freq < min_hits:
        return None, freq

    return best_phrase, freq


def run_phrase_assistant(
    s1_raw: List[str],
    s2_raw: List[str],
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    batch_size: int,
    dry_run: bool,
    phrase_len: int = 5,
    min_phrase_freq: int = 2,
    min_hits: int = 1,
    max_phrases: int = 20,
) -> List[CorrRule]:
    if dry_run:
        log.warning("[assist-phrases] Ignoré en mode dry-run.")
        return []

    log.info("[assist-phrases] Démarrage de l'assistant de corrélation (phrases)…")

    D0 = embed_in_batches(
        s2_raw,
        role="doc",
        batch_size=batch_size,
        emb_client=emb_client,
        log=log,
        dry_run=False,
    )
    Q0 = embed_in_batches(
        s1_raw,
        role="query",
        batch_size=batch_size,
        emb_client=emb_client,
        log=log,
        dry_run=False,
    )
    best_idx0, _ = cosine_two_phase_global(Q0, D0, threshold=0.0, log=log)

    candidates = find_candidate_phrases_correlated(
        s1_raw=s1_raw,
        best_idx=best_idx0,
        min_phrase_freq=min_phrase_freq,
        max_phrases=max_phrases,
        n=phrase_len,
    )
    if not candidates:
        print(f"\n[assist-phrases] Aucun bout de phrase candidat détecté (n={phrase_len}).")
        return []

    print("\n=== Assistant de corrélation sur bouts de phrase ===")
    print(f"(n-grammes de {phrase_len} mots, sélectionnés dans le fichier 1)")

    raw_suggestions: List[Tuple[str, str, int]] = []
    for ph, global_freq in candidates:
        tgt_ph, hits = suggest_target_phrase(
            ph,
            s1_raw=s1_raw,
            s2_raw=s2_raw,
            best_idx=best_idx0,
            n=phrase_len,
            min_hits=min_hits,
        )
        if not tgt_ph:
            continue
        raw_suggestions.append((ph, tgt_ph, hits))

    if not raw_suggestions:
        print("\n[assist-phrases] Aucune corrélation de phrase suffisamment stable détectée.")
        return []

    print("\nPropositions initiales (avant édition) :")
    for idx, (src, tgt, hits) in enumerate(raw_suggestions, start=1):
        print(f"  [{idx}] '{src}' -> '{tgt}' (observé ≈{hits} fois)")

    final_pairs = interactive_edit_suggestions(raw_suggestions)

    if not final_pairs:
        return []

    corr_rules: List[CorrRule] = []
    for src, tgt in final_pairs:
        pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
        corr_rules.append((pattern, tgt))
        log.info(f"[assist-phrases] Règle ajoutée: '{src}' -> '{tgt}'")

    return corr_rules


# === Helpers parsing réponse reranker ===
# (Reranker conservé pour compat, mais non utilisé dans le matching global)
def _extract_scores_from_results(res: List[dict], n_candidates: int) -> List[float]:
    if not isinstance(res, list):
        raise RuntimeError("Champ results/result non liste")

    if all(isinstance(r, dict) for r in res):
        if all("index" in r for r in res):
            scores = [0.0] * n_candidates
            for r in res:
                idx = int(r.get("index", -1))
                if 0 <= idx < n_candidates:
                    scores[idx] = float(r.get("score", 0.0))
            return scores

        if len(res) == n_candidates:
            return [float(r.get("score", 0.0)) for r in res]

    raise RuntimeError("Format results/result non exploitable")


def call_reranker_http(
    query: str, candidates: List[str], http_client: httpx.Client, log: Logger
) -> List[float]:
    if not candidates:
        return []

    data = {
        "query": query,
        "documents": candidates,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if RERANK_API_KEY and RERANK_API_KEY != "EMPTY":
        headers["Authorization"] = f"Bearer {RERANK_API_KEY}"

    endpoint = "rerank"
    url = f"{RERANK_API_BASE.rstrip('/')}/{endpoint}"

    log.debug(f"[rerank] URL finale = {url}")
    log.debug(f"[rerank] Payload envoyé = {data}")

    resp = http_client.post(url, headers=headers, json=data, timeout=60.0)
    resp.raise_for_status()

    payload = resp.json()
    log.debug(f"[rerank] RAW PAYLOAD = {payload!r}")

    if not isinstance(payload, dict):
        raise RuntimeError("Payload reranker non dict")

    if "scores" in payload:
        scores = payload["scores"]
        if len(scores) != len(candidates):
            raise RuntimeError("Longueur scores != documents")
        return [float(s) for s in scores]

    if "results" in payload:
        return _extract_scores_from_results(payload["results"], len(candidates))

    if "result" in payload:
        return _extract_scores_from_results(payload["result"], len(candidates))

    if "data" in payload:
        return _extract_scores_from_results(payload["data"], len(candidates))

    raise RuntimeError("Format de réponse reranker non reconnu")


# =========================
#  Interface CLI simple
# =========================
def ask_str(prompt_text: str, default: Optional[str] = None) -> str:
    txt = input(
        f"{prompt_text}{' [' + default + ']' if default else ''}: "
    ).strip()
    return txt or (default or "")


def ask_int(prompt_text: str, default: Optional[int] = None) -> int:
    while True:
        raw = input(
            f"{prompt_text}{' [' + str(default) + ']' if default is not None else ''}: "
        ).strip()
        raw = raw or (str(default) if default is not None else "")
        try:
            v = int(raw)
            if v >= 1:
                return v
        except Exception:
            pass
        print("Veuillez entrer un entier >= 1.")


def interactive_collect(use_dialog: bool = False):
    print("=== Mode interactif ===")
    file1 = ask_str("Sélectionnez FICHIER 1", "L:\\Test\\CLasseur1.xlsx")
    sheet1 = ask_str("Nom de FEUILLE 1", "Feuil1")
    col1 = ask_int("Numéro de COLONNE 1 (1=A)", 1)
    file2 = ask_str("Sélectionnez FICHIER 2", "L:\\Test\\CLasseur2.xlsx")
    sheet2 = ask_str("Nom de FEUILLE 2", "Feuil1")
    col2 = ask_int("Numéro de COLONNE 2 (1=A)", 1)
    threshold = float(ask_str("Seuil cosine", "0.78") or "0.78")

    match_mode = ask_str("Méthode de matching (full/approx)", "full").strip().lower()
    if match_mode not in ("full", "approx"):
        match_mode = "full"
    topk = 10
    if match_mode == "approx":
        topk = ask_int("k pour matching approx top-k (par source)", 10)

    out_prefix = ask_str("Dossier de sortie", ".")
    limit = ask_str("Limiter à N lignes (vide = pas de limite)", "")
    limit = int(limit) if (limit and limit.isdigit()) else None

    corr_answer = ask_str("Activer la corrélation forcée ? (y/N)", "N").strip().lower()
    force_corr = corr_answer in ("y", "yes", "o", "oui")

    assist_answer = ask_str(
        "Activer la corrélation automatique sur mots simples ? (y/N)",
        "N"
    ).strip().lower()
    assist_corr = assist_answer in ("y", "yes", "o", "oui")

    assist_phr_answer = ask_str(
        "Activer la corrélation sur bouts de phrase (5 mots) ? (y/N)",
        "N"
    ).strip().lower()
    assist_phrases = assist_phr_answer in ("y", "yes", "o", "oui")

    phrase_len = 5
    if assist_phrases:
        try:
            phrase_len = ask_int("Longueur des bouts de phrase (en mots)", 5)
        except Exception:
            phrase_len = 5

    return {
        "file1": file1,
        "sheet1": sheet1,
        "col1": col1,
        "file2": file2,
        "sheet2": sheet2,
        "col2": col2,
        "threshold": threshold,
        "out_prefix": out_prefix,
        "limit": limit,
        "force_corr": force_corr,
        "assist_corr": assist_corr,
        "assist_phrases": assist_phrases,
        "phrase_len": phrase_len,
        "match_mode": match_mode,
        "topk": topk,
    }


# =========================
#  Main
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Comparaison sémantique via Snowflake embeddings (spy-mode)."
    )
    ap.add_argument("--file1")
    ap.add_argument("--sheet1")
    ap.add_argument("--col1", type=int)
    ap.add_argument("--file2")
    ap.add_argument("--sheet2")
    ap.add_argument("--col2", type=int)
    ap.add_argument("--threshold", type=float, default=0.78)
    ap.add_argument("--out-prefix", default=".")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--ask", action="store_true")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre de lignes par base (debug)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne pas appeler l'API, embeddings aléatoires (debug)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Logs verbeux (ou DEBUG=1)",
    )
    ap.add_argument(
        "--rerank-k",
        type=int,
        default=0,
        help="(Conservé pour compat) — non utilisé avec le matching global.",
    )
    ap.add_argument(
        "--near-band",
        type=float,
        default=0.05,
        help="(Conservé pour compat) — non utilisé avec le matching global.",
    )
    ap.add_argument(
        "--llm-equivalent",
        action="store_true",
        help="Utiliser DALLEM pour détecter des exigences equivalentes",
    )
    ap.add_argument(
        "--llm-max",
        type=int,
        default=200,
        help="Nombre maximum de lignes à analyser avec le LLM (matches + missmatchs)",
    )
    ap.add_argument(
        "--force-corr",
        action="store_true",
        help="Activer la corrélation forcée (normalisation texte avant embeddings)",
    )
    ap.add_argument(
        "--assist-corr",
        action="store_true",
        help="Scanner les bases et proposer des corrélations automatiques sur mots simples",
    )
    ap.add_argument(
        "--assist-phrases",
        action="store_true",
        help="Proposer des corrélations sur des bouts de phrase (n-grammes).",
    )
    ap.add_argument(
        "--phrase-len",
        type=int,
        default=5,
        help="Longueur des bouts de phrase (en mots) pour --assist-phrases (défaut: 5).",
    )
    ap.add_argument(
        "--match-mode",
        choices=["full", "approx"],
        default="full",
        help="Méthode de matching: 'full' (matrice complète) ou 'approx' (top-k par source).",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Valeur de k pour --match-mode=approx (top-k par source).",
    )

    args = ap.parse_args()

    debug_flag = args.debug or (
        os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")
    )
    log = make_logger(debug_flag)

    if (args.assist_corr or args.assist_phrases) and not args.force_corr:
        log.info("[assist] Assistant de corrélation activé -> forçage de --force-corr.")
        args.force_corr = True

    log.info("=== Configuration ===")
    log.info(f"SNOWFLAKE_API_BASE = {SNOWFLAKE_API_BASE}")
    log.info(f"DALLEM_API_BASE    = {DALLEM_API_BASE}")
    log.info(f"RERANK_API_BASE    = {RERANK_API_BASE}")
    log.info(f"VERIFY_SSL         = {VERIFY_SSL}")
    log.info(f"EMBED_MODEL        = {EMBED_MODEL}")
    log.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    log.info(
        "API_KEYS           = snowflake={} | dallem={} | rerank={}".format(
            _mask(SNOWFLAKE_API_KEY),
            _mask(DALLEM_API_KEY),
            _mask(RERANK_API_KEY),
        )
    )
    log.info(f"RERANK_K           = {args.rerank_k} (non utilisé en mode global)")
    log.info(f"NEAR_BAND          = {args.near_band} (non utilisé en mode global)")
    log.info(f"LLM_equivalent (CLI)      = {args.llm_equivalent}")
    log.info(f"LLM_ANTAGONISM_ENABLED    = {LLM_ANTAGONISM_ENABLED}")
    log.info(f"LLM_MAX            = {args.llm_max}")
    log.info(f"FORCE_CORR         = {args.force_corr}")
    log.info(f"ASSIST_CORR        = {args.assist_corr}")
    log.info(f"ASSIST_PHRASES     = {args.assist_phrases}")
    log.info(f"PHRASE_LEN         = {args.phrase_len}")
    log.info(f"MATCH_MODE         = {args.match_mode}")
    log.info(f"TOPK               = {args.topk}")

    llm_enabled = LLM_ANTAGONISM_ENABLED or args.llm_equivalent
    log.info(f"[llm] Activation effective equivalent = {llm_enabled}")

    if not SNOWFLAKE_API_KEY and not args.dry_run:
        log.error("SNOWFLAKE_API_KEY manquant (ou utilisez --dry-run).")
        sys.exit(1)

    try:
        _http_client = httpx.Client(
            verify=VERIFY_SSL, timeout=httpx.Timeout(300.0)
        )
    except Exception as e:
        log.error(f"[httpx] Erreur création client: {e}")
        log.debug(traceback.format_exc())
        sys.exit(2)

    llm_client: Optional[DirectOpenAILLM] = None
    if llm_enabled:
        if not DALLEM_API_KEY or DALLEM_API_KEY == "toto":
            log.error("DALLEM_API_KEY manquant ou de test. Impossible d'utiliser le LLM.")
        else:
            llm_client = DirectOpenAILLM(
                model=LLM_MODEL,
                api_key=DALLEM_API_KEY,
                base_url=DALLEM_API_BASE,
                http_client=_http_client,
                logger=log,
            )
            log.info(f"[llm] DALLEM activé (modèle={LLM_MODEL})")
            try:
                eq_test, expl_test = llm_client.analyse_equivalence(
                    "Vitesse maximale 250 kt.", "La vitesse maximale est limitée à 250 kt."
                )
                log.info(f"[llm] TEST: equiv={eq_test} | expl={expl_test}")
            except Exception as e:
                log.error(f"[llm] Erreur lors de l'appel de TEST LLM: {e}")

    if args.interactive or not all(
        [args.file1, args.sheet1, args.col1, args.file2, args.sheet2, args.col2]
    ):
        ia = interactive_collect(use_dialog=args.ask)
        args.file1 = ia["file1"]
        args.sheet1 = ia["sheet1"]
        args.col1 = ia["col1"]
        args.file2 = ia["file2"]
        args.sheet2 = ia["sheet2"]
        args.col2 = ia["col2"]
        args.threshold = ia["threshold"]
        args.out_prefix = ia["out_prefix"]
        args.limit = ia["limit"]
        args.match_mode = ia["match_mode"]
        args.topk = ia["topk"]

        args.force_corr = ia["force_corr"]
        if ia["assist_corr"]:
            args.assist_corr = True
        if ia["assist_phrases"]:
            args.assist_phrases = True
            args.phrase_len = ia["phrase_len"]

        if (args.assist_corr or args.assist_phrases) and not args.force_corr:
            log.info("[assist] Activation interactive -> corrélation forcée activée.")
            args.force_corr = True

    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=_http_client,
        role_prefix=False,
        logger=log,
    )

    try:
        s1 = read_excel_col(args.file1, args.sheet1, args.col1, log)
        s2 = read_excel_col(args.file2, args.sheet2, args.col2, log)
    except Exception:
        log.error("Arrêt: erreur lecture Excel.")
        sys.exit(3)

    if args.limit:
        log.info(f"[limit] on tronque à {args.limit} lignes par base (debug)")
        s1 = s1[: args.limit]
        s2 = s2[: args.limit]

    if not s1 or not s2:
        log.error("Colonnes vides après nettoyage.")
        sys.exit(4)

    s1_raw = list(s1)
    s2_raw = list(s2)

    corr_rules: List[CorrRule] = []

    # Assistant mots simples
    if args.force_corr and args.assist_corr:
        try:
            auto_rules_words = run_corr_assistant(
                s1_raw=s1_raw,
                s2_raw=s2_raw,
                emb_client=emb_client,
                log=log,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                min_term_freq=3,
                min_hits=2,
                max_terms=20,
            )
            corr_rules.extend(auto_rules_words)
        except Exception as e:
            log.error(f"[assist-corr] Erreur lors de l'assistant de corrélation (mots): {e}")
            log.debug(traceback.format_exc())

    # Assistant bouts de phrase
    if args.force_corr and args.assist_phrases:
        try:
            phrase_rules = run_phrase_assistant(
                s1_raw=s1_raw,
                s2_raw=s2_raw,
                emb_client=emb_client,
                log=log,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                phrase_len=args.phrase_len,
                min_phrase_freq=2,
                min_hits=1,
                max_phrases=20,
            )
            if phrase_rules:
                corr_rules.extend(phrase_rules)
        except Exception as e:
            log.error(f"[assist-phrases] Erreur lors de l'assistant de corrélation (phrases): {e}")
            log.debug(traceback.format_exc())

    # Corrélation manuelle + défaut
    if args.force_corr:
        try:
            manual_rules = collect_corr_rules_from_user(log)
            if manual_rules:
                corr_rules.extend(manual_rules)

            if not corr_rules:
                corr_rules = build_default_corr_rules(log)

            s1 = [apply_corr_rules(t, corr_rules) for t in s1]
            s2 = [apply_corr_rules(t, corr_rules) for t in s2]

            log.debug("[corr] Sample B1 avant/après :")
            for i in range(min(3, len(s1_raw))):
                log.debug(f"  RAW: {s1_raw[i]!r}")
                log.debug(f"  COR: {s1[i]!r}")
        except Exception as e:
            log.error(f"[corr] Erreur lors de l'application des règles de corrélation: {e}")
            log.debug(traceback.format_exc())
            s1 = s1_raw
            s2 = s2_raw

    log.info(f"[bases] Base1={len(s1)} | Base2={len(s2)}")
    log.debug(f"[bases] B1 sample (corrigé ou non): {s1[:3]}")
    log.debug(f"[bases] B2 sample (corrigé ou non): {s2[:3]}")

    try:
        D = embed_in_batches(
            s2,
            role="doc",
            batch_size=args.batch_size,
            emb_client=emb_client,
            log=log,
            dry_run=args.dry_run,
        )
        Q = embed_in_batches(
            s1,
            role="query",
            batch_size=args.batch_size,
            emb_client=emb_client,
            log=log,
            dry_run=args.dry_run,
        )
    except Exception:
        log.error("Arrêt: échec génération embeddings.")
        sys.exit(5)

    # Matching global 2 phases (rerank ignoré, mais accès conservés)
    if args.rerank_k and args.rerank_k > 0:
        log.warning(
            "[pipeline] Rerank demandé, mais non supporté avec le matching global 2 phases. "
            "On utilise uniquement le cosinus."
        )

    try:
        if getattr(args, "match_mode", "full") == "approx":
            k = max(1, int(getattr(args, "topk", 10) or 10))
            log.info(f"[pipeline] Matching approx top-k activé (k={k})")
            pairs_topk = cosine_topk_pairs(Q, D, k=k, log=log)
            best_idx, best_val = cosine_two_phase_global_from_pairs(
                Q.shape[0],
                D.shape[0],
                pairs_topk,
                threshold=args.threshold,
                log=log,
            )
        else:
            log.info("[pipeline] Matching matrice complète (mode 'full')")
            best_idx, best_val = cosine_two_phase_global(
                Q,
                D,
                threshold=args.threshold,
                log=log,
            )
    except Exception as e:
        log.error(f"[cosine] Erreur: {e}")
        log.debug(traceback.format_exc())
        sys.exit(6)

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
        if tgt_idx is not None and score >= args.threshold:
            matches_above.append(row)
        else:
            under.append(row)

    # =========================
    #  Analyse LLM (matches + missmatchs, promotion)
    # =========================
    if llm_client is not None:
        budget = max(0, int(args.llm_max))
        used_above = 0

        # 1) LLM sur les matches au-dessus du seuil
        if matches_above and budget > 0:
            n_above = min(len(matches_above), budget)
            used_above = n_above
            log.info(
                f"[llm] Analyse des équivalences sur {n_above} matches (sur {len(matches_above)} au total)"
            )
            for k_idx in range(n_above):
                row = matches_above[k_idx]
                src = row["source"]
                tgt = row["target"]
                antago, expl = llm_client.analyse_equivalence(src, tgt)
                row["équivalence"] = antago
                row["commentaire"] = expl
                row.setdefault("promu_par_llm", False)

            for k_idx in range(n_above, len(matches_above)):
                row = matches_above[k_idx]
                row.setdefault("équivalence", None)
                row.setdefault("commentaire", "Non analysé (limite llm_max atteinte)")
                row.setdefault("promu_par_llm", False)

        # 2) LLM sur les missmatchs (sous seuil) avec le budget restant
        remaining = max(0, budget - used_above)
        if under and remaining > 0:
            n_under = min(len(under), remaining)
            log.info(
                f"[llm] Analyse des missmatchs sur {n_under} lignes (sur {len(under)} au total)"
            )

            for idx in range(n_under):
                row = under[idx]
                src = row["source"]
                tgt = row["target"]
                antago, expl = llm_client.analyse_equivalence(src, tgt)
                row["équivalence"] = antago
                row["commentaire"] = expl
                row.setdefault("promu_par_llm", False)

            for idx in range(n_under, len(under)):
                row = under[idx]
                row.setdefault("équivalence", None)
                row.setdefault("commentaire", "Non analysé (LLM non appelé sur cette ligne)")
                row.setdefault("promu_par_llm", False)

            # Promotion des missmatchs marqués True, en respectant l'unicité côté cible
            used_targets = {
                r.get("tgt_index")
                for r in matches_above
                if r.get("tgt_index") is not None
            }
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
            log.info(f"[llm] Promotion de {promoted_count} missmatch(s) en matches (unicité conservée).")

    # =========================
    #  Export
    # =========================
    out_dir = args.out_prefix
    os.makedirs(out_dir, exist_ok=True)
    path_matches = os.path.join(out_dir, "matches.xlsx")
    path_under = os.path.join(out_dir, "under_threshold.xlsx")

    try:
        pd.DataFrame(matches_above).to_excel(
            path_matches, index=False, engine="xlsxwriter"
        )
        pd.DataFrame(under).to_excel(
            path_under, index=False, engine="xlsxwriter"
        )
    except Exception as e:
        log.error(f"[export] Erreur écriture Excel: {e}")
        log.debug(traceback.format_exc())
        sys.exit(7)

    log.info(
        f"✅ Terminé. "
        f"Matches (>= {args.threshold} ou promus LLM, unicité globale) : {len(matches_above)} | "
        f"Sous seuil restants : {len(under)}"
    )
    log.info(f"- {path_matches}")
    log.info(f"- {path_under}")
    log.info("📄 Log détaillé: compareur_debug.log")


if __name__ == "__main__":
    main()