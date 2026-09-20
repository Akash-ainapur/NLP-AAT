# Fake News Detection Using a Pretrained Transformer Classifier

> Scaffold. Sections marked **[WRITE]** are yours — the rest are factual
> descriptions of the system that you should still read and edit into your
> own words before submitting.

---

## 1. Introduction

**[WRITE]** — What misinformation is, why automated screening is of interest,
and what this project actually does: integrate an existing fine-tuned
transformer into a web service that classifies pasted article text.

State plainly that no model was trained here. Examiners respect this more
than a vague claim of having "built a model".

## 2. Problem Statement

**[WRITE]** — Given the raw text of a news article, produce a binary label
(fake / real) with a confidence score, exposed through a web interface.

Note the scope limit: the system classifies *text as written*. It does not
retrieve evidence, check claims against sources, or verify facts.

## 3. Related Work

**[WRITE]** — Three or four short paragraphs. Suggested points:

- Early approaches: hand-engineered features (n-grams, TF-IDF) with classical
  classifiers such as Naive Bayes, SVM, logistic regression.
- Transformer era: BERT (Devlin et al., 2019) and fine-tuning pretrained
  language models for text classification.
- Common datasets: ISOT fake-and-real-news, LIAR (Wang, 2017), FakeNewsNet.
- Known criticism: reported accuracies above 99% on single datasets are widely
  attributed to dataset artefacts rather than genuine fact-verification
  ability. Worth one honest sentence — it sets up Section 8.

## 4. Methodology

### 4.1 Model

The system uses `jy46604790/Fake-News-Bert-Detect`, a publicly available
checkpoint on the Hugging Face Hub. It is a BERT-based sequence classification
model fine-tuned for binary fake-news detection. It emits two labels:

| Raw label | Meaning |
|---|---|
| `LABEL_0` | Fake |
| `LABEL_1` | Real |

along with a softmax confidence score.

### 4.2 BERT in brief

**[WRITE]** — One paragraph: bidirectional transformer encoder, pretrained on
masked-language-modelling and next-sentence prediction, fine-tuned downstream
by adding a classification head over the `[CLS]` token representation.

### 4.3 Input handling

Input is truncated to 450 words before inference. The checkpoint truncates
around 500 words internally, so trimming first avoids transmitting tokens that
would be discarded.

## 5. System Architecture

```
   Browser (single page)
        |  POST /analyze  {"text": "..."}
        v
   FastAPI backend
        |
        |--1--> Hugging Face Inference API  ---> label + score
        |            (on any failure)
        '--2--> local transformers pipeline ---> label + score
        |
        v
   {"verdict", "confidence", "served_by", "latency_ms"}
```

**[WRITE]** — Redraw this as a proper figure for the report.

### 5.1 Why there are two inference paths

The hosted free tier cold-starts idle models and caps daily requests, so a
hosted-only design can fail at exactly the wrong moment. Every request tries
the hosted API first and transparently falls back to a locally cached copy of
the same model. The response records which path served it.

### 5.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | serves the single-page interface |
| `GET` | `/samples` | returns the six built-in demo articles |
| `POST` | `/analyze` | classifies one article |

## 6. Implementation

**[WRITE]** — Walk through `app/classifier.py` and `app/main.py`. Mention:
FastAPI with Pydantic request validation, a minimum input length of 40
characters, the background warm-up thread that pre-loads the local model at
startup, and the single-page frontend, which keeps the pasted article on
screen and shows the result alongside it rather than replacing it.

## 7. Results

**[WRITE]** — Run `python evaluate.py --limit 100` and paste the output.

Report: number of articles, accuracy, the per-class precision / recall / F1
table, and the confusion matrix. State which dataset you evaluated on and how
many articles — a metric without its sample size is not a result.

Observed latency: roughly 200 ms per article once the local model is resident;
the first call after startup costs a one-time model load of about 60 seconds,
which is why the server warms the model in the background at boot.

## 8. Limitations

This section is the most important one in the report. Be specific.

**It classifies style, not truth.** The model has no access to external
evidence. It cannot check whether a claim is accurate — it can only recognise
that a passage resembles text labelled fake in its training data. A carefully
written false story and a badly written true one are both likely to be
misread.

**It inherits its training distribution.** The real articles in the datasets
these models are trained on come largely from wire services, so neutral,
formal, wire-style prose is what the model has learned "real" looks like.

This was observed directly while building the demo set. The model confidently
and correctly flagged sensationalist fabricated articles, and correctly passed
wire-style political and economic reporting. But it labelled two ordinary,
neutrally written real articles — one about a court hearing, one about a
university study — as fake at over 98% confidence. Neither was misleading;
they simply did not match the register of the training data.

**It is confident when it is wrong.** The scores above were not hedged
predictions near 50%. Softmax confidence should not be read as probability of
correctness.

**Language and domain.** English only, and general news only.

**[WRITE]** — Add a sentence on what you would do to address these given more
time (evaluating across multiple datasets, calibration, evidence retrieval).

## 9. Conclusion

**[WRITE]** — What was built, what it does, what the measured performance was,
and an honest statement of where it should and should not be relied on.

## References

**[WRITE]** — In your institution's required format. At minimum:

1. Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers
   for Language Understanding.* NAACL-HLT, 2019.
2. Wang, W. Y. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake
   News Detection.* ACL, 2017.
3. Ahmed, H., Traore, I., Saad, S. *Detection of Online Fake News Using
   N-Gram Analysis and Machine Learning Techniques.* ISDDC, 2017. (ISOT dataset)
4. Hugging Face model card: `jy46604790/Fake-News-Bert-Detect`.
