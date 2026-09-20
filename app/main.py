import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import classifier

def _setup_logging():
    """Own our loggers explicitly.

    uvicorn configures root logging before this module is imported, which
    makes logging.basicConfig a silent no-op and leaves root at WARNING - so
    our INFO lines (including the "model ready" signal the README tells you
    to wait for) would never print.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    for name in ("app", "classifier"):
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


@asynccontextmanager
async def lifespan(_app):
    # Loading the local model takes ~60s the first time. Do it in the
    # background at boot so it is never paid for mid-demo, and keep the
    # server responsive while it happens.
    def warm():
        try:
            classifier.warm_local()
            log.info("local fallback model ready")
        except Exception as exc:
            log.warning("could not warm local model: %s", exc)

    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="Fake News Detection", lifespan=lifespan)


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
    return classifier.analyze(text)
