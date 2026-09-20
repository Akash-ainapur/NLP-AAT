"""Score the classifier on a labelled sample so the report has a results table.

Evaluation runs against the local model rather than the hosted API: a few
hundred calls would be slow and would eat the free-tier daily quota.

Usage:
    python evaluate.py                 # uses data/True.csv + data/Fake.csv (ISOT)
    python evaluate.py --limit 100
    python evaluate.py --csv mydata.csv    # needs `text` and `label` columns
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app.classifier import MODEL_ID, LABEL_MAP, truncate_words

DATA_DIR = Path("data")
ISOT_REAL = DATA_DIR / "True.csv"
ISOT_FAKE = DATA_DIR / "Fake.csv"

DOWNLOAD_HINT = f"""
No evaluation data found.

Download the ISOT fake-news dataset from Kaggle:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Unzip it so you have:
    {ISOT_REAL}
    {ISOT_FAKE}

Or pass your own labelled file:
    python evaluate.py --csv yourfile.csv   (columns: text, label)
"""


def load_isot(limit):
    real = pd.read_csv(ISOT_REAL)
    fake = pd.read_csv(ISOT_FAKE)
    per_class = max(limit // 2, 1)

    # Sample rather than take the head: the CSVs are ordered by topic and date,
    # so the first N rows are not representative.
    real = real.sample(n=min(per_class, len(real)), random_state=42)
    fake = fake.sample(n=min(per_class, len(fake)), random_state=42)

    real = pd.DataFrame({"text": real["text"], "label": "REAL"})
    fake = pd.DataFrame({"text": fake["text"], "label": "FAKE"})
    return pd.concat([real, fake]).sample(frac=1, random_state=42).reset_index(drop=True)


def load_csv(path, limit):
    frame = pd.read_csv(path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        sys.exit(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    frame["label"] = frame["label"].str.upper()
    return frame.sample(n=min(limit, len(frame)), random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="articles to score")
    parser.add_argument("--csv", type=Path, help="labelled CSV with text,label columns")
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    args = parser.parse_args()

    if args.csv:
        data = load_csv(args.csv, args.limit)
    elif ISOT_REAL.exists() and ISOT_FAKE.exists():
        data = load_isot(args.limit)
    else:
        sys.exit(DOWNLOAD_HINT)

    from transformers import pipeline

    print(f"loading {MODEL_ID} ...")
    clf = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, truncation=True)

    texts = [truncate_words(str(t)) for t in data["text"]]
    print(f"scoring {len(texts)} articles ...")
    predictions = clf(texts, batch_size=8)

    data["predicted"] = [LABEL_MAP.get(p["label"], p["label"]) for p in predictions]
    data["confidence"] = [round(p["score"], 4) for p in predictions]

    accuracy = (data["predicted"] == data["label"]).mean()

    print(f"\nModel:    {MODEL_ID}")
    print(f"Articles: {len(data)}")
    print(f"Accuracy: {accuracy:.3f}\n")
    print(classification_report(data["label"], data["predicted"], digits=3))
    print("Confusion matrix (rows = actual FAKE, REAL / cols = predicted):")
    print(confusion_matrix(data["label"], data["predicted"], labels=["FAKE", "REAL"]))

    data.to_csv(args.out, index=False)
    print(f"\nper-article results written to {args.out}")


if __name__ == "__main__":
    main()
