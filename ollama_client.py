# ollama_client.py
import requests
import json
from typing import Any
from config import BASE_URL, OLLAMA_GEN_PATH, OLLAMA_EMB_PATH
import logging

logger = logging.getLogger(__name__)

def ollama_generate(prompt: str, model: str | None = None, max_tokens: int = 512, temperature: float = 0.5) -> str:
    url = BASE_URL.rstrip("/") + OLLAMA_GEN_PATH
    payload = {
        "model": model, 
        "prompt": prompt,
        "stream": False, 
        "options":{
            "max_tokens": max_tokens, 
            "temperature": temperature,
            },
        }
    logger.info(f"Ollama generate request -> model={model}, max_tokens={max_tokens}, temp={temperature}")
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        logger.warning("Non-JSON response from Ollama generate")
        return resp.text
    if isinstance(data, dict):
        for k in ("text","result","generation","response","output"):
            if k in data:
                return data[k] if isinstance(data[k], str) else json.dumps(data[k], ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)

def ollama_embed(text: str, model: str | None = None) -> list[float]:
    logger.debug(f"Ollama embedding request -> model={model}, text_len={len(text)}")
    m = model or None
    base = BASE_URL.rstrip("/")
    urls = [base + OLLAMA_EMB_PATH, base + "/embeddings", base + "/api/embeddings"]
    payload = {"model": m, "prompt": text}
    last_err = None
    for url in urls:
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                for k in ("embedding","embeddings","vector","vectors"):
                    if k in data:
                        return data[k]
            if isinstance(data, list) and isinstance(data[0], dict) and "embedding" in data[0]:
                return data[0]["embedding"]
        except Exception as e:
            logger.warning(f"Ollama embed failed at {url}: {e}")
            last_err = e
            continue
    raise RuntimeError(f"All embedding endpoints failed. Last error: {last_err}")
