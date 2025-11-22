"""
Module de gestion des modèles locaux pour le mode hors ligne.
Supporte les modèles LLM et d'embeddings stockés localement.
"""

import os
import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger("offline_models")


# =========================
#  Configuration des modèles
# =========================

# Modèles LLM disponibles
AVAILABLE_LLM_MODELS = {
    "qwen": "D:\\IA Test\\models\\Qwen\\Qwen2.5-3B-Instruct",
    "mistral": "D:\\IA Test\\models\\mistralai\\Mistral-7B-Instruct-v0.3",
}

# Modèle d'embedding par défaut
DEFAULT_EMBEDDING = "D:\\IA Test\\models\\BAAI\\bge-m3"


class OfflineEmbeddingModel:
    """
    Client d'embeddings local utilisant sentence-transformers ou transformers.
    """

    def __init__(self, model_path: str, logger: Optional[logging.Logger] = None):
        self.model_path = model_path
        self.log = logger or logging.getLogger("offline_embeddings")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Charge le modèle d'embeddings."""
        try:
            # Essayer d'abord sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.log.info(f"[offline-emb] Chargement du modèle depuis {self.model_path}")
                self.model = SentenceTransformer(self.model_path)
                self.backend = "sentence-transformers"
                self.log.info("[offline-emb] Modèle chargé avec sentence-transformers")
            except ImportError:
                # Fallback sur transformers
                from transformers import AutoTokenizer, AutoModel
                import torch

                self.log.info(f"[offline-emb] Chargement du modèle avec transformers depuis {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
                self.backend = "transformers"
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.log.info(f"[offline-emb] Modèle chargé avec transformers sur {self.device}")

        except Exception as e:
            self.log.error(f"[offline-emb] Erreur lors du chargement du modèle: {e}")
            raise

    def _embed_transformers(self, texts: List[str]) -> np.ndarray:
        """Génère des embeddings avec transformers."""
        import torch

        self.log.debug(f"[offline-emb] Génération de {len(texts)} embeddings avec transformers")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour des documents."""
        if not texts:
            return []

        try:
            if self.backend == "sentence-transformers":
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                embeddings = self._embed_transformers(texts)
                # Normalisation
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-12)

            self.log.debug(f"[offline-emb] Généré {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
            return embeddings.tolist()

        except Exception as e:
            self.log.error(f"[offline-emb] Erreur lors de la génération d'embeddings: {e}")
            raise

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour des requêtes."""
        return self.embed_documents(texts)


class OfflineLLMModel:
    """
    Client LLM local utilisant transformers ou llama.cpp.
    """

    def __init__(self, model_path: str, logger: Optional[logging.Logger] = None):
        self.model_path = model_path
        self.log = logger or logging.getLogger("offline_llm")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Charge le modèle LLM."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.log.info(f"[offline-llm] Chargement du modèle depuis {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available():
                self.model.to(self.device)

            self.log.info(f"[offline-llm] Modèle chargé sur {self.device}")

        except Exception as e:
            self.log.error(f"[offline-llm] Erreur lors du chargement du modèle: {e}")
            raise

    def analyse_equivalence(self, text1: str, text2: str) -> Tuple[Optional[bool], str]:
        """
        Analyse si deux textes sont équivalents.
        Retourne (équivalent: bool, explication: str)
        """
        try:
            import torch

            system_msg = (
                "Tu es un expert en exigences aéronautiques. "
                "On te donne deux exigences techniques. "
                "Tu dois répondre si elles expriment la même chose (ou quasi) en 15 mots. "
            )

            user_msg = (
                f"Exigence A :\n{text1}\n\n"
                f"Exigence B :\n{text2}\n\n"
                "Les deux exigences expriment-elles la même chose ou quasiment, "
                "ta reponse doit impérativement commencer par TRUE ou FALSE ? "
            )

            # Format selon le modèle
            if "qwen" in self.model_path.lower():
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            elif "mistral" in self.model_path.lower():
                prompt = f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
            else:
                prompt = f"{system_msg}\n\n{user_msg}\n\nRéponse: "

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = response.strip()

            self.log.debug(f"[offline-llm] Réponse brute: {response!r}")

            # Analyse de la réponse
            tokens = response.split()
            first = tokens[0].lower() if tokens else ""

            if "true" in first:
                equivalent = True
            elif "false" in first:
                equivalent = False
            else:
                equivalent = None

            return equivalent, response

        except Exception as e:
            self.log.error(f"[offline-llm] Erreur lors de l'analyse d'équivalence: {e}")
            return None, f"ERREUR: {str(e)}"


def get_offline_embedding_client(model_key: str = "bge-m3", logger: Optional[logging.Logger] = None):
    """
    Crée un client d'embeddings offline.
    """
    if model_key == "bge-m3":
        model_path = DEFAULT_EMBEDDING
    else:
        raise ValueError(f"Modèle d'embedding non reconnu: {model_key}")

    return OfflineEmbeddingModel(model_path, logger)


def get_offline_llm_client(model_key: str, logger: Optional[logging.Logger] = None):
    """
    Crée un client LLM offline.
    """
    if model_key not in AVAILABLE_LLM_MODELS:
        raise ValueError(f"Modèle LLM non reconnu: {model_key}. Disponibles: {list(AVAILABLE_LLM_MODELS.keys())}")

    model_path = AVAILABLE_LLM_MODELS[model_key]
    return OfflineLLMModel(model_path, logger)


# Adaptateur pour compatibilité avec le code existant
class OfflineEmbeddingsAdapter:
    """
    Adaptateur pour rendre OfflineEmbeddingModel compatible avec DirectOpenAIEmbeddings.
    """

    def __init__(self, model_key: str = "bge-m3", logger: Optional[logging.Logger] = None):
        self.client = get_offline_embedding_client(model_key, logger)
        self.log = logger or logging.getLogger("offline_adapter")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_queries(texts)


class OfflineLLMAdapter:
    """
    Adaptateur pour rendre OfflineLLMModel compatible avec DirectOpenAILLM.
    """

    def __init__(self, model_key: str, logger: Optional[logging.Logger] = None):
        self.client = get_offline_llm_client(model_key, logger)
        self.log = logger or logging.getLogger("offline_llm_adapter")

    def analyse_equivalence(self, text1: str, text2: str) -> Tuple[Optional[bool], str]:
        return self.client.analyse_equivalence(text1, text2)
