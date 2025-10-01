# embedder_ollama.py
import numpy as np
from typing import List
from ollama_client import ollama_embed
from config import OLLAMA_MODEL_EMB
import logging

logger = logging.getLogger(__name__)

class OllamaEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or OLLAMA_MODEL_EMB
        self._dim = None
        logger.info(f"OllamaEmbedder initialized with model={self.model_name}")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Embedding batch of {len(texts)} texts with {self.model_name}")
        embs = []
        for t in texts:
            logger.debug(f"Embedding text snippet: {t[:50]}...")
            vec = ollama_embed(t, model=self.model_name)
            embs.append(vec)
        arr = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        arr = arr / norms
        if self._dim is None and arr.shape[1]:
            self._dim = arr.shape[1]
            logger.info(f"Embedding dimension set to {self._dim}")
        return arr

    def embed(self, text: str) -> np.ndarray:
        logger.debug(f"Embedding single text: {text[:80]}...")
        return self.embed_batch([text])[0]

    def get_dim(self) -> int:
        if self._dim is None:
            logger.info("Determining embedding dimension with test call...")
            v = self.embed("test")
            self._dim = v.shape[0]
        return self._dim
