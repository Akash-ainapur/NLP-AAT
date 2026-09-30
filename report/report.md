# Fake News Detection Using an LLM via OpenRouter

> Scaffold. Sections marked **[WRITE]** are yours — the rest are factual
> descriptions of the system that you should still read and edit into your
> own words before submitting.

---

## 1. Introduction

**[WRITE]** — What misinformation is, why automated screening is of interest,
and what this project actually does: integrate a hosted large language model into a web service that classifies
pasted article text.

State plainly that no model was trained here. Examiners respect this more
than a vague claim of having "built a model".

## 2. Problem Statement

**[WRITE]** — Given the raw text of a news article, produce a label
(real / fake / uncertain) with a confidence score, exposed through a web interface.

Note the scope limit: the system classifies *text as written*. Fact-checking is
delegated to the LLM and its optional web search; the system itself keeps no
evidence store.

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

The first version used a BERT checkpoint fine-tuned on the ISOT dataset. It
labelled neutral, recent real news as fake because it had learned writing
style from old wire-service text, not facts. The system now sends the article
to Google's Gemini API (default `gemini-2.5-flash`, configurable) with Grounding
with Google Search enabled, so the model can check the main claim against live
search results, which a fixed classifier cannot do. On the free tier this
allows up to 500 grounded requests per day for the 2.5 Flash family.

### 4.2 Prompt and output

The prompt is deliberately short. It asks for one of REAL, FAKE or UNCERTAIN
and an integer confidence from 0 to 100, and nothing else. Temperature is 0.
The reply is parsed and validated; malformed output becomes UNCERTAIN with
confidence 0 instead of an error.

### 4.3 Input handling

Input is truncated to 1,500 words before the call to bound cost and latency.

## 5. System Architecture

```
   Browser (single page)
        |  POST /analyze  {"text": "..."}
        v
   FastAPI backend (app/main.py)
        |
        v
   FakeNewsDetector (app/detector.py)
        |  Settings -> GeminiClient -> Gemini API (+ Google Search grounding)
        v
   {"verdict", "confidence", "model", "latency_ms"}
```

**[WRITE]** — Redraw this as a proper figure for the report.

### 5.1 Classes

| Class | Role |
|---|---|
| `Settings` | reads API key, model, search flag from `.env` |
| `GeminiClient` | one HTTP call with timeout and one retry |
| `FakeNewsDetector` | truncates, prompts, parses and validates the reply |
| `Prediction` | Pydantic schema for verdict and confidence |
| `DetectorError` | API failures, returned to the UI as HTTP 502 |

### 5.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | serves the single-page interface |
| `GET` | `/samples` | returns the built-in demo articles |
| `POST` | `/analyze` | classifies one article |

## 6. Implementation

**[WRITE]** — Walk through `app/detector.py` and `app/main.py`. Mention:
FastAPI with Pydantic request validation, a minimum input length of 40
characters, the short JSON-only prompt, the UNCERTAIN outcome, and the
single-page frontend, which keeps the pasted article on screen and shows the
result alongside it.

## 7. Results

**[WRITE]** — Run `python evaluate.py --limit 20` and paste the output.

Report the number of articles, accuracy, the UNCERTAIN count and the confusion
matrix. Also compare against the earlier BERT result on the same recent-news
samples. A metric without its sample size is not a result.

Latency is typically a few seconds per article, longer with search on.

## 8. Limitations

**It can be wrong and sound sure.** The confidence is the model's own
estimate, not a measured accuracy.

**Without search it cannot know recent events.** Its knowledge stops at a
training cutoff, so grounding should stay on for current news.

**It needs internet and an API key.** The free tier has daily request limits.

**Answers can vary** between runs and models, although temperature 0 reduces
this.

**Language and domain.** English and general news only.

**[WRITE]** — Add what you would do given more time (evaluation across several
datasets, comparing models, showing the cited sources).

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
4. OpenRouter API documentation, openrouter.ai/docs.
