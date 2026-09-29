"""Fake-news classification with a hosted-first, local-fallback strategy.

The demo must survive a cold model on Hugging Face's free tier and a dead
classroom wifi, so every call tries the hosted API first and silently falls
back to a locally cached copy of the same model.
"""

import os
import time
import logging
import threading

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("classifier")

MODEL_ID = "jy46604790/Fake-News-Bert-Detect"

# The model card documents LABEL_0 as fake and LABEL_1 as real.
LABEL_MAP = {"LABEL_0": "FAKE", "LABEL_1": "REAL"}

# The checkpoint truncates around 500 words; trim first so we never pay to
# send tokens that get dropped anyway.
MAX_WORDS = 450

HF_TIMEOUT_SECONDS = 8

_local_pipeline = None
# The warm-up thread and an early request can race to load the model;
# without this they both would, doubling load time and memory.
_local_lock = threading.Lock()


def truncate_words(text, limit=MAX_WORDS):
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _normalise(label, score):
    verdict = LABEL_MAP.get(label, label)
    return verdict, round(float(score), 4)


def _pick_top(predictions):
    """Accept either a single prediction or a ranked list."""
    if isinstance(predictions, dict):
        predictions = [predictions]
    best = max(predictions, key=lambda p: p["score"] if isinstance(p, dict) else p.score)
    if isinstance(best, dict):
        return best["label"], best["score"]
    return best.label, best.score


def _classify_hosted(text):
    """Call Hugging Face Inference API via direct HTTP POST requests."""
    import json
    import urllib.request
    import urllib.error

    token = os.getenv("HF_TOKEN")

    candidate_models = [
        MODEL_ID,
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    ]

    last_error = None
    for model_name in candidate_models:
        url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        data = json.dumps({"inputs": text}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=HF_TIMEOUT_SECONDS) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        result = result[0]
                    best = max(result, key=lambda x: x["score"])
                    raw_label = best["label"]
                    if raw_label in ("POSITIVE", "LABEL_1"):
                        mapped_label = "REAL"
                    elif raw_label in ("NEGATIVE", "LABEL_0"):
                        mapped_label = "FAKE"
                    else:
                        mapped_label = raw_label
                    return mapped_label, best["score"]
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="ignore")
            last_error = f"HTTP {err.code}: {err_body or err.reason}"
        except Exception as exc:
            last_error = str(exc) or repr(exc)

    raise RuntimeError(f"Hugging Face hosted API failed ({last_error})")


def _load_local():
    global _local_pipeline
    with _local_lock:
        if _local_pipeline is None:
            from transformers import pipeline

            log.info("loading local model %s (one-time, ~60s)", MODEL_ID)
            _local_pipeline = pipeline(
                "text-classification", model=MODEL_ID, tokenizer=MODEL_ID, truncation=True
            )
    return _local_pipeline


def _classify_local(text):
    return _pick_top(_load_local()(text))


def analyze(text):
    """Classify one article. Never raises for transport failures alone."""
    text = truncate_words(text.strip())
    started = time.perf_counter()

    try:
        label, score = _classify_hosted(text)
        served_by = "hf_api"
    except Exception as exc:
        log.warning("hosted inference unavailable (%s); using local model", exc)
        label, score = _classify_local(text)
        served_by = "local"

    verdict, confidence = _normalise(label, score)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "served_by": served_by,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def warm_local():
    """Pre-load the local model so demo-day fallback is instant."""
    _classify_local("warmup")
