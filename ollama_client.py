import requests
import json
from typing import Any
from config import BASE_URL, OLLAMA_GEN_PATH, OLLAMA_EMB_PATH

def ollama_generate(prompt: str, model: str | None = None, max_tokens: int = 512, temperature: float = 0.0, timeout: int = 60) -> str:
    url = BASE_URL.rstrip("/") + OLLAMA_GEN_PATH
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        return resp.text
    # tolerant parsing
    if isinstance(data, dict):
        for k in ("text","result","generation","response","output"):
            if k in data:
                v = data[k]
                if isinstance(v, str):
                    return v
                if isinstance(v, list):
                    return "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in v])
                if isinstance(v, dict) and "text" in v:
                    return v["text"]
    return json.dumps(data, ensure_ascii=False)

def ollama_embed(text: str, model: str | None = None, timeout: int = 60) -> list[float]:
    """
    Call embedding endpoint. Different Ollama versions use different paths.
    Try embedded path then fallback to common alternatives.
    """
    m = model or None
    # try configured path
    base = BASE_URL.rstrip("/")
    urls = [base + OLLAMA_EMB_PATH, base + "/embeddings", base + "/api/embeddings"]
    payload = {"model": m, "prompt": text}
    last_err = None
    for url in urls:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # tolerant parse
            if isinstance(data, dict):
                for k in ("embedding","embeddings","vector","vectors"):
                    if k in data:
                        emb = data[k]
                        if isinstance(emb, list):
                            return emb
                        if isinstance(emb, dict) and "data" in emb:
                            return emb["data"]
            # some APIs return list
            if isinstance(data, list) and isinstance(data[0], dict) and "embedding" in data[0]:
                return data[0]["embedding"]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All embedding endpoints failed. Last error: {last_err}")
