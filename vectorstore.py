# vectorstore.py
import faiss
import numpy as np
import sqlite3
import json
from config import FAISS_INDEX_PATH, METADATA_DB

class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index_path = FAISS_INDEX_PATH
        self.meta_conn = sqlite3.connect(METADATA_DB, check_same_thread=False)
        self._ensure_meta_table()
        self._load_or_create_index()

    def _ensure_meta_table(self):
        c = self.meta_conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            idx INTEGER PRIMARY KEY,
            doc_id TEXT,
            meeting_id TEXT,
            meeting_type TEXT,
            chunk_text TEXT,
            metadata TEXT
        )""")
        self.meta_conn.commit()

    def _load_or_create_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
        except Exception:
            self.index = faiss.IndexFlatIP(self.dim)
        c = self.meta_conn.cursor()
        c.execute("SELECT COUNT(1) FROM docs")
        self._count = c.fetchone()[0] or 0

    def save_index(self):
        faiss.write_index(self.index, self.index_path)

    def add_documents(self, docs, embeddings: np.ndarray):
        n = embeddings.shape[0]
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings)
        c = self.meta_conn.cursor()
        for i, doc in enumerate(docs):
            idx = self._count + i
            c.execute("INSERT INTO docs (idx, doc_id, meeting_id, meeting_type, chunk_text, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                      (idx, doc["doc_id"], doc.get("meeting_id"), doc.get("meeting_type"), doc.get("chunk_text"), json.dumps(doc.get("metadata", {}), ensure_ascii=False)))
        self.meta_conn.commit()
        self._count += n
        self.save_index()

    def query(self, q_embedding: np.ndarray, top_k: int = 5, meeting_type: str | None = None):
        if q_embedding.ndim == 1:
            q = q_embedding.reshape(1, -1)
        else:
            q = q_embedding
        D, I = self.index.search(q, top_k)
        c = self.meta_conn.cursor()
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            c.execute("SELECT doc_id, meeting_id, meeting_type, chunk_text, metadata FROM docs WHERE idx = ?", (idx,))
            row = c.fetchone()
            if not row:
                continue
            doc_id, meeting_id, mt, chunk_text, metadata = row
            if meeting_type and mt != meeting_type:
                continue
            try:
                meta = json.loads(metadata)
            except Exception:
                meta = {}
            results.append({"doc_id": doc_id, "meeting_id": meeting_id, "meeting_type": mt, "chunk_text": chunk_text, "metadata": meta, "score": float(score)})
        return results
