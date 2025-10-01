# pipeline.py
import uuid, json, time, logging
from embedder_ollama import OllamaEmbedder
from vectorstore import FaissVectorStore
from kag import KnowledgeStore
from extractor import simple_local_extract, ollama_extract_facts
from cache import make_cache
from ollama_client import ollama_generate
from config import CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_MODEL_GEN, CACHE_TTL
import numpy as np
from tqdm import tqdm
from profile_builder import ProfileBuilder

logger = logging.getLogger(__name__)

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
        logger.info("Initializing Pipeline...")
        self.embedder = OllamaEmbedder()
        self.dim = self.embedder.get_dim()
        self.vs = FaissVectorStore(dim=self.dim)
        self.kag = KnowledgeStore()
        self.cache = make_cache()
        self.use_ollama_extractor = use_ollama_extractor
        self.profile_builder = ProfileBuilder(self)
        logger.info("Pipeline initialized successfully.")

    def ingest_meeting(self, meeting_id: str, transcript: str,
                       meeting_type: str | None = None, metadata: dict | None = None):
        start = time.time()
        logger.info(f"[INGEST] Start ingesting meeting {meeting_id} (type={meeting_type})")

        # Запись метаданных встречи
        self.kag.upsert_meeting(meeting_id, meeting_type=meeting_type, metadata=metadata)
        if not meeting_type:
            mt = self.classify_meeting_type(transcript)
            meeting_type = mt or "unknown"
            self.kag.upsert_meeting(meeting_id, meeting_type=meeting_type)
            logger.info(f"[INGEST] Auto-classified meeting {meeting_id} as {meeting_type}")

        # Извлечение фактов/участников
        logger.info(f"[INGEST] Extracting facts/participants for meeting {meeting_id}")
        if self.use_ollama_extractor:
            ext = ollama_extract_facts(transcript)
        else:
            ext = simple_local_extract(transcript)

        for p in ext.get("participants", []):
            self.kag.upsert_participant(p, metadata={"source_meeting": meeting_id})
        logger.info(f"[INGEST] Added {len(ext.get('participants', []))} participants")

        for f in ext.get("facts", []):
            self.kag.add_fact(
                meeting_id=meeting_id,
                subject=f.get("subject", "unknown"),
                predicate=f.get("predicate", ""),
                obj=f.get("object", ""),
                confidence=f.get("confidence", 0.6),
                source="extractor"
            )
        logger.info(f"[INGEST] Added {len(ext.get('facts', []))} facts")

        # Чанкинг
        logger.info(f"[INGEST] Splitting transcript into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        chunks = chunk_text(transcript)
        logger.info(f"[INGEST] Transcript split into {len(chunks)} chunks")

        # Эмбеддинги
        texts = [c for c in chunks]
        logger.info(f"[INGEST] Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts)
        logger.info(f"[INGEST] Finished embedding {len(texts)} chunks")

        # Добавление в векторку
        docs = []
        for c in chunks:
            doc_id = str(uuid.uuid4())
            docs.append({
                "doc_id": doc_id,
                "meeting_id": meeting_id,
                "meeting_type": meeting_type,
                "chunk_text": c,
                "metadata": metadata or {}
            })
        self.vs.add_documents(docs, embeddings)
        logger.info(f"[INGEST] Added {len(docs)} documents to vectorstore")

        elapsed = time.time() - start
        logger.info(f"[INGEST] Finished ingesting meeting {meeting_id} in {elapsed:.2f}s")

    def classify_meeting_type(self, transcript: str) -> str | None:
        logger.debug("[CLASSIFY] Classifying meeting type")
        prompt = ("К какому типу относится эта встреча. Выбери одну из категорий: "
                  "sales, support, standup, management, strategy, interview, other. "
                  "Верни только название категории.\n\nТекст:\n" + transcript + "\n\nКатегория:")
        resp = ollama_generate(prompt=prompt)
        if not resp:
            return None
        return resp.strip().split()[0].lower()

    def retrieve_candidates(self, query: str, top_k_per_type: int = 5):
        logger.info(f"[RETRIEVE] Retrieving candidates for query: {query}")
        cache_key = f"types_for:{query}"
        cached = self.cache.get(cache_key)
        if cached:
            candidates = json.loads(cached)
            logger.info("[RETRIEVE] Loaded candidates from cache")
        else:
            candidates = self.kag.get_meeting_type_candidates_for_query(query)
            self.cache.set(cache_key, json.dumps(candidates), ttl=600)
            logger.info(f"[RETRIEVE] Computed candidates: {candidates}")

        q_emb = self.embedder.embed(query)
        results = []
        if candidates:
            for t in candidates:
                logger.debug(f"[RETRIEVE] Querying vectorstore for type {t}")
                results.extend(self.vs.query(q_emb, top_k=top_k_per_type, meeting_type=t))
        if not results:
            logger.debug("[RETRIEVE] No candidates by type, querying all")
            results = self.vs.query(q_emb, top_k=top_k_per_type)

        uniq, seen = [], set()
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["doc_id"] not in seen:
                uniq.append(r)
                seen.add(r["doc_id"])
        logger.info(f"[RETRIEVE] Retrieved {len(uniq[:top_k_per_type*3])} unique candidates")
        return uniq[:top_k_per_type*3]

    def answer_query(self, query: str, top_k: int = 6,
                     use_facts: bool = True, cache_ttl: int = CACHE_TTL):
        start = time.time()
        logger.info(f"[ANSWER] Answering query: {query}")

        cache_key = f"answer:{query}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("[ANSWER] Loaded answer from cache")
            return json.loads(cached)

        candidates = self.retrieve_candidates(query, top_k_per_type=top_k//2)
        facts = []
        if use_facts:
            kws = [w for w in query.split() if len(w) > 3]
            facts = self.kag.query_facts(keywords=kws, limit=20)
            logger.info(f"[ANSWER] Retrieved {len(facts)} relevant facts")

        logger.info(f"[ANSWER] Building prompt with {len(candidates)} docs and {len(facts)} facts")
        system = ("You are an assistant that must answer concisely in Russian "
                  "using ONLY the provided contexts and facts. "
                  "If information is missing, say 'Не хватает данных'. "
                  "Include provenance.")
        docs_text = "\n\n".join(
            [f"[doc {d['doc_id']} meeting={d['meeting_id']} type={d['meeting_type']} score={d['score']:.3f}]\n{d['chunk_text']}"
             for d in candidates[:top_k]]
        )
        facts_text = "\n".join(
            [f"[fact {f.id}] {f.subject} | {f.predicate} | {f.object} (conf={f.confidence:.2f})"
             for f in facts[:20]]
        )

        prompt = f"{system}\n\nКонтекст:\n{docs_text}\n\nФакты:\n{facts_text}\n\nВопрос: {query}\nОтвет:"
        answer = ollama_generate(prompt=prompt)

        out = {
            "answer": answer,
            "provenance": {
                "docs": [d['doc_id'] for d in candidates[:top_k]],
                "facts": [f.id for f in facts[:20]]
            }
        }
        self.cache.set(cache_key, json.dumps(out, ensure_ascii=False), ttl=cache_ttl)

        elapsed = time.time() - start
        logger.info(f"[ANSWER] Finished answering query in {elapsed:.2f}s")
        return out

    def build_profile_from_files(self, name: str, meeting_files: list,
                                 model: str | None = None, output_dir: str = "outputs"):
        logger.info(f"[PROFILE] Building profile for {name} from {len(meeting_files)} files")
        start = time.time()

        meetings = []
        provenance = []
        for path in meeting_files:
            logger.info(f"[PROFILE] Reading file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            base = path.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
            if "_" in base:
                meeting_type, meeting_id = base.split("_", 1)
            else:
                meeting_id = base
                meeting_type = None
            meetings.append({
                "meeting_id": meeting_id,
                "text": txt,
                "meeting_type": meeting_type
            })
            provenance.append({"meeting_id": meeting_id, "file": path})
            logger.info(f"[PROFILE] Loaded meeting {meeting_id} (type={meeting_type}) from {path}")

        res = self.profile_builder.build_profile(
            name=name,
            meetings=meetings,
            model=model,
            output_dir=output_dir
        )

        elapsed = time.time() - start
        logger.info(f"[PROFILE] Profile for {name} built in {elapsed:.2f}s")
        return res
