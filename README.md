# Fake News Detection

A web service that judges whether a news article is real or fake and says how
sure it is, using Google's Gemini API with Google Search grounding.

- **Model:** `gemini-2.5-flash` (configurable)
- **Backend:** FastAPI
- **Frontend:** one page, vanilla HTML/CSS/JS, no build step

The earlier version used a BERT classifier that judged writing style and
flagged neutral recent news as fake. It was replaced by an LLM that reasons
about the claim and checks it against live Google Search results.

## Setup

```bash
pip install -r requirements-local.txt   # requirements.txt is enough to just run the app
cp .env.example .env        # then paste your Gemini key into .env
```

Get a free key at <https://aistudio.google.com/apikey>. Settings in `.env`:

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | required |
| `GEMINI_MODEL` | default `gemini-2.5-flash` |
| `USE_WEB_SEARCH` | `true` turns on Google Search grounding (needed for today's news) |

Google Search grounding is free on the free tier only for the 2.5 Flash and
Flash-Lite models (up to 500 requests a day). Gemini 3.x models need a paid
tier for it.

## Run

```bash
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

## Output

`POST /analyze` returns only:

```json
{"verdict": "REAL", "confidence": 87, "model": "gemini-2.5-flash", "latency_ms": 2400}
```

`verdict` is `REAL`, `FAKE` or `UNCERTAIN` (hard to tell). If the API fails
the endpoint returns HTTP 502 with a readable message.

## Evaluation

```bash
python evaluate.py --limit 100
```

Prints accuracy, precision, recall, F1 and a confusion matrix, and writes
per-article results to `results.csv` for the report.

Needs a labelled dataset. Download the
[ISOT fake-and-real-news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
and unzip it to `data/True.csv` and `data/Fake.csv`, or supply your own CSV
with `text` and `label` columns via `--csv`.

Each article is one API call, so keep `--limit` small to stay inside the free quota. `UNCERTAIN` is
reported separately and counts as not matching the label.

## Known limitations

- The confidence is the model's own estimate, not a measured accuracy.
- With `USE_WEB_SEARCH=false` the model cannot know events after its training
  cutoff, so keep it on for current news.
- Needs internet and an API key. Read the result as a signal, not a verdict.

## Deploy to Vercel

The repository is ready for [Vercel](https://vercel.com) (`vercel.json` and
`api/index.py`).

1. Push to GitHub and import the project into Vercel.
2. In Project Settings -> Environment Variables add `GEMINI_API_KEY`
   (and optionally `GEMINI_MODEL`, `USE_WEB_SEARCH`).

Serverless functions have a time limit; a Gemini call takes a few seconds,
longer with search on.

## Layout

```
api/index.py          Vercel serverless entrypoint
vercel.json           Vercel route configuration
app/detector.py       Settings, GeminiClient, FakeNewsDetector
app/main.py           FastAPI routes
app/static/index.html the whole frontend
samples/examples.json eight demo articles
evaluate.py           metrics for the report
```
