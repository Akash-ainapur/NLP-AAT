"""Fake-news detection with Gemini and Google Search grounding.

The model is asked for one thing only: a verdict and how sure it is.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

log = logging.getLogger("detector")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = (
    "You are a fake-news detector. Read the article and decide if it is REAL news, "
    "FAKE news, or UNCERTAIN if you cannot tell.\n"
    "Judge sourcing, sensationalism, internal consistency and plausibility. "
    "If web search is available, check the main claim.\n"
    'Reply with ONLY JSON: {"verdict":"REAL"|"FAKE"|"UNCERTAIN","confidence":<integer 0-100>}\n'
    "Confidence is how sure you are in that verdict. "
    "Use UNCERTAIN with low confidence when evidence is insufficient."
)


class DetectorError(Exception):
    """The LLM call failed in a way the caller should report, not hide."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = "gemini-3.5-flash-lite"
    use_web_search: bool = True
    timeout_seconds: float = 45.0
    max_words: int = 1500

    @classmethod
    def from_env(cls):
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise DetectorError("GEMINI_API_KEY is not set. Add it to .env.")
        return cls(
            api_key=key,
            model=os.getenv("GEMINI_MODEL", cls.model).strip(),
            use_web_search=os.getenv("USE_WEB_SEARCH", "true").strip().lower()
            in ("1", "true", "yes", "on"),
        )


class Prediction(BaseModel):
    verdict: Literal["REAL", "FAKE", "UNCERTAIN"]
    confidence: int = Field(ge=0, le=100)


class GeminiClient:
    def __init__(self, settings):
        self.settings = settings

    @property
    def model_id(self):
        return self.settings.model

    def complete(self, system, user):
        """Return (reply_text, grounded). Falls back to no search if search is refused."""
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            # JSON mode cannot be combined with the search tool, so the JSON
            # shape is enforced by the prompt and _parse instead.
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1024},
        }
        if self.settings.use_web_search:
            try:
                payload["tools"] = [{"google_search": {}}]
                return self._post(payload, attempts=1), True
            except DetectorError as exc:
                # The free tier rejects grounding on newer models with 429/400.
                log.warning("search grounding unavailable (%s); retrying without it", exc)
                del payload["tools"]
        return self._post(payload), False

    def _post(self, payload, attempts=2):
        headers = {"x-goog-api-key": self.settings.api_key}
        url = API_URL.format(model=self.settings.model)

        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers,
                    timeout=self.settings.timeout_seconds,
                )
            except httpx.HTTPError as exc:
                last_error = f"network error: {exc}"
            else:
                if response.status_code == 200:
                    return self._content(response)
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                # Retry only transient failures; a bad key or bad request won't fix itself.
                if response.status_code not in (408, 429, 503) and response.status_code < 500:
                    break
            log.warning("Gemini attempt %d failed (%s)", attempt, last_error)
            if attempt < attempts:
                time.sleep(1.0)

        raise DetectorError(f"Gemini request failed - {last_error}")

    @staticmethod
    def _content(response):
        try:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (ValueError, KeyError, IndexError, TypeError):
            # Safety blocks and empty candidates land here; treat as "no answer".
            log.warning("Gemini returned no usable candidate: %s", response.text[:200])
            return ""


class FakeNewsDetector:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        self.client = GeminiClient(self.settings)

    def analyze(self, text):
        started = time.perf_counter()
        raw, grounded = self.client.complete(SYSTEM_PROMPT, self._truncate(text.strip()))
        prediction = self._parse(raw)
        return {
            "verdict": prediction.verdict,
            "confidence": prediction.confidence,
            "model": self.client.model_id,
            "grounded": grounded,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    def _truncate(self, text):
        words = text.split()
        if len(words) <= self.settings.max_words:
            return text
        return " ".join(words[: self.settings.max_words])

    @staticmethod
    def _parse(raw):
        """Turn the model's reply into a Prediction; unusable output becomes UNCERTAIN."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate code fences / chatter
        try:
            data = json.loads(match.group(0)) if match else {}
            data["verdict"] = str(data.get("verdict", "")).upper()
            data["confidence"] = max(0, min(100, int(round(float(data.get("confidence", 0))))))
            return Prediction(**data)
        except (ValueError, TypeError, ValidationError):
            log.warning("unparseable model output: %r", raw[:200])
            return Prediction(verdict="UNCERTAIN", confidence=0)
