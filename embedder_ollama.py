import numpy as np
from typing import List
from ollama_client import ollama_embed
from config import OLLAMA_MODEL_EMB

class OllamaEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or OLLAMA_MODEL_EMB
        # dimension is unknown until first call

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        embs = []
        for t in texts:
            vec = ollama_embed(t, model=self.model_name)
            embs.append(vec)
        arr = np.array(embs, dtype=np.float32)
        # optionally normalize to unit length to use inner product as cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        arr = arr / norms
        return arr

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def get_dim(self) -> int:
        v = self.embed("test")
        return v.shape[0]
