# extractor.py
import re
from typing import Dict, List
from ollama_client import ollama_generate

def simple_local_extract(transcript: str) -> dict:
    participants = set()
    facts = []
    for m in re.findall(r"[\w\.-]+@[\w\.-]+", transcript):
        participants.add(m)
    for m in re.findall(r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\b", transcript):
        if len(m.split()) <= 3:
            participants.add(m.strip())
    for sent in re.split(r"[.\n]", transcript):
        s = sent.strip()
        if not s:
            continue
        lowered = s.lower()
        if any(k in lowered for k in ("должен","сделать","ответствен","назначен","сделать")):
            subj = re.search(r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\b", s)
            subj = subj.group(1) if subj else "unknown"
            facts.append({"subject": subj, "predicate": "action", "object": s.strip(), "confidence": 0.6})
    return {"participants": list(participants), "facts": facts}

def ollama_extract_facts(transcript: str, model: str = None) -> dict:
    prompt = (
        "Extract participants and simple facts from the following Russian meeting transcript. "
        "Return JSON with keys: participants (list of names/emails), facts (list of {subject,predicate,object,confidence}). "
        "Be concise and return only valid JSON.\n\nTranscript:\n" + transcript + "\n\nJSON:"
    )
    raw = ollama_generate(prompt=prompt)
    try:
        import json, re
        m = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.S)
        if m:
            j = json.loads(m.group(1))
            parts = j.get("participants", []) if isinstance(j, dict) else []
            facts = j.get("facts", []) if isinstance(j, dict) else []
            return {"participants": parts, "facts": facts}
    except Exception:
        pass
    return simple_local_extract(transcript)
