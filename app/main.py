import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.detector import DetectorError, FakeNewsDetector

def _setup_logging():
    """Own our loggers explicitly.

    uvicorn configures root logging before this module is imported, which
    makes logging.basicConfig a silent no-op and leaves root at WARNING - so
    our INFO lines (including the "model ready" signal the README tells you
    to wait for) would never print.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    for name in ("app", "detector"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers = [handler]
        logger.propagate = False


_setup_logging()
log = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_FILE = BASE_DIR.parent / "samples" / "examples.json"

MIN_CHARS = 40


detector = None


def get_detector():
    # Built lazily so a missing key surfaces as a readable API error, not a crash at boot.
    global detector
    if detector is None:
        detector = FakeNewsDetector()
    return detector


app = FastAPI(title="Fake News Detection")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw article text to classify")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/samples")
def samples():
    return json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))


@app.post("/analyze")
def analyze(payload: AnalyzeRequest):
    text = payload.text.strip()
    if len(text) < MIN_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {MIN_CHARS} characters to classify.",
        )
    try:
        return get_detector().analyze(text)
    except DetectorError as exc:
        log.error("analysis failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
