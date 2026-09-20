# Fake News Detection

A web service that classifies a news article as likely fake or likely real,
built on a pretrained BERT classifier from Hugging Face.

- **Model:** [`jy46604790/Fake-News-Bert-Detect`](https://huggingface.co/jy46604790/Fake-News-Bert-Detect)
- **Backend:** FastAPI
- **Frontend:** one page, vanilla HTML/CSS/JS, no build step

No model was trained for this project. An existing fine-tuned checkpoint is
integrated into a web application.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your token into .env
```

Get a token at <https://huggingface.co/settings/tokens> — a **fine-grained**
token with the *"Make calls to Inference Providers"* permission.

The token is optional. Without it the app falls back to running the model
locally, which works exactly the same but downloads ~440 MB on first use.

## Run

```bash
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

## How it handles failure

Every request tries the hosted Hugging Face API first and falls back to a
local copy of the same model if that fails for any reason — cold start, rate
limit, expired token, no internet. The response says which path served it,
and the page shows it in small text under the result.

The local model is loaded in a background thread when the server boots, so
the one-time ~60 s load never lands in the middle of a demo.

## Before demo day

1. Start the server once **with internet** so the model is downloaded and cached.
2. Wait for `local fallback model ready` in the log.
3. Turn the wifi off and run a sample. It must still work.

If step 3 works, nothing on the network can break the demo.

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

Evaluation deliberately runs against the local model: a few hundred hosted
calls would be slow and would consume the free-tier daily quota.

## Known limitation

The model was fine-tuned on a specific English news dataset whose real
articles come largely from wire services. It is therefore sensitive to
**writing style** rather than to whether a claim is actually true.

In testing, it correctly flagged sensationalist fake articles at high
confidence and correctly passed wire-service-style political and economic
reporting — but misclassified a neutrally written article about a court
hearing, and a neutral article about a university study, as fake. Both were
ordinary real news that simply did not match the style of its training data.

It should be read as a signal, not a verdict. See `report/report.md`.

## Layout

```
app/classifier.py     hosted call, local fallback, label mapping
app/main.py           FastAPI routes
app/static/index.html the whole frontend
samples/examples.json six demo articles
evaluate.py           metrics for the report
```
