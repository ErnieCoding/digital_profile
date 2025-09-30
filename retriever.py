from embedder_ollama import Embedder
from vectorstore import FaissVectorStore
from kag import KnowledgeStore

class Retriever:
    def __init__(self, embedder: Embedder, vectorstore: FaissVectorStore, kag: KnowledgeStore):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.kag = kag

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embedder.embed([query])[0]
        docs = self.vectorstore.query(q_emb, top_k=top_k)
        # also pull facts by keywords
        keywords = [w for w in query.split() if len(w) > 3]
        facts = self.kag.search_facts_by_keywords(keywords, limit=10) if keywords else []
        return {"docs": docs, "facts": facts}
