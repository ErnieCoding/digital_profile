import uuid, json
from embedder_ollama import OllamaEmbedder
from vectorstore import FaissVectorStore
from kag import KnowledgeStore
from extractor import simple_local_extract, ollama_extract_facts
from cache import make_cache
from ollama_client import ollama_generate
from config import CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_MODEL_GEN, CACHE_TTL
import numpy as np
from tqdm import tqdm

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out

class Pipeline:
    def __init__(self, use_ollama_extractor: bool = False):
        self.embedder = OllamaEmbedder()
        self.dim = self.embedder.get_dim()
        self.vs = FaissVectorStore(dim=self.dim)
        self.kag = KnowledgeStore()
        self.cache = make_cache()
        self.use_ollama_extractor = use_ollama_extractor

    def ingest_meeting(self, meeting_id: str, transcript: str, meeting_type: str | None = None, metadata: dict | None = None):
        # 1) ensure meeting record (may be updated)
        self.kag.upsert_meeting(meeting_id, meeting_type=meeting_type, metadata=metadata)
        # 2) classify meeting_type if not provided (lightweight call to Ollama)
        if not meeting_type:
            mt = self.classify_meeting_type(transcript)
            meeting_type = mt or "unknown"
            self.kag.upsert_meeting(meeting_id, meeting_type=meeting_type)
        # 3) extract facts & participants
        if self.use_ollama_extractor:
            ext = ollama_extract_facts(transcript)
        else:
            ext = simple_local_extract(transcript)
        for p in ext.get("participants", []):
            self.kag.upsert_participant(p, metadata={"source_meeting": meeting_id})
        for f in ext.get("facts", []):
            self.kag.add_fact(meeting_id=meeting_id, subject=f.get("subject","unknown"), predicate=f.get("predicate",""), obj=f.get("object",""), confidence=f.get("confidence",0.6), source="extractor")
        # 4) chunk + embed + index (with meeting metadata)
        chunks = chunk_text(transcript)
        docs = []
        texts = []
        for c in chunks:
            doc_id = str(uuid.uuid4())
            docs.append({"doc_id": doc_id, "meeting_id": meeting_id, "meeting_type": meeting_type, "chunk_text": c, "metadata": metadata or {}})
            texts.append(c)
        embeddings = self.embedder.embed_batch(texts)
        self.vs.add_documents(docs, embeddings)

    def classify_meeting_type(self, transcript: str) -> str | None:
        prompt = ("К какому типу относится эта встреча. Выбери одну из категорий: sales, support, standup, management, strategy, interview, other. "
                  "Верни только название категории без описания.\n\nТекст встречи:\n" + transcript + "\n\nКатегория:")
        resp = ollama_generate(prompt, model=OLLAMA_MODEL_GEN, max_tokens=20, temperature=0.0)
        if not resp:
            return None
        
        return resp.strip().split()[0].lower()

    def retrieve_candidates(self, query: str, top_k_per_type: int = 5):
        """
        1) ask KAG for likely meeting_type candidates
        2) if candidates exist, search per-type and aggregate
        3) else search globally
        """
        cache_key = f"types_for:{query}"
        cached = self.cache.get(cache_key)
        if cached:
            candidates = json.loads(cached)
        else:
            candidates = self.kag.get_meeting_type_candidates_for_query(query)
            self.cache.set(cache_key, json.dumps(candidates), ttl=600)
        results = []
        # embed query once
        q_emb = self.embedder.embed(query)
        if candidates:
            for t in candidates:
                docs = self.vs.query(q_emb, top_k=top_k_per_type, meeting_type=t)
                results.extend(docs)
            # if not enough, search globally
            if len(results) < top_k_per_type:
                results.extend(self.vs.query(q_emb, top_k=top_k_per_type))
        else:
            results = self.vs.query(q_emb, top_k=top_k_per_type)
        # deduplicate by doc_id and sort by score descending
        seen = set()
        uniq = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["doc_id"] in seen:
                continue
            seen.add(r["doc_id"])
            uniq.append(r)
        return uniq[:top_k_per_type*3]

    def answer_query(self, query: str, top_k: int = 6, use_facts: bool = True, cache_ttl: int = CACHE_TTL):
        cache_key = f"answer:{query}"
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)

        candidates = self.retrieve_candidates(query, top_k_per_type=top_k//2)
        # fetch KAG facts relevant
        facts = []
        if use_facts:
            kws = [w for w in query.split() if len(w)>3]
            facts = self.kag.query_facts(keywords=kws, limit=20)

        # build prompt
        system = ("You are an assistant that must answer concisely in Russian using ONLY the provided contexts and facts. "
                  "If information is missing, respond 'Не хватает данных'. "
                  "Include provenance: meeting_id or fact ids.")
        docs_text = "\n\n".join([f"[doc {d['doc_id']} meeting={d['meeting_id']} type={d['meeting_type']} score={d['score']:.3f}]\n{d['chunk_text']}" for d in candidates[:top_k]])
        facts_text = "\n".join([f"[fact {f.id}] {f.subject} | {f.predicate} | {f.object} (conf={f.confidence:.2f})" for f in facts[:20]])
        prompt = f"{system}\n\nКонтекст:\n{docs_text}\n\nФакты:\n{facts_text}\n\nВопрос: {query}\nОтвет:"
        answer = ollama_generate(prompt, model=OLLAMA_MODEL_GEN, max_tokens=512, temperature=0.0)
        out = {"answer": answer, "provenance": {"docs": [d['doc_id'] for d in candidates[:top_k]], "facts": [f.id for f in facts[:20]]}}
        self.cache.set(cache_key, json.dumps(out, ensure_ascii=False), ttl=cache_ttl)
        return out
