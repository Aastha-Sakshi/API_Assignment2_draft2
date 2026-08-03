# AI Recruitment Assistant
### CCZG506 — Assignment II (API-driven Cloud Native Solutions)

**Domain:** HR / Recruitment
**Categories:** Computer Vision + Natural Language Processing
**Unified objective:** screen one candidate against one job description

---

## 1. How this maps to the assignment

| # | Requirement | Where it is satisfied |
|---|---|---|
| 1 | Select a domain | HR / Recruitment |
| 2 | Two categories | Computer Vision + NLP |
| 3 | ≥5 sub-tasks | **6** — see table below |
| 4 | Identify models | `GET /model-registry` returns every model, task and category |
| 5 | Cohesive, unified objective | `POST /screen-candidate` chains the sub-tasks into one screening result |
| 6 | API-driven, interactive, demonstrable | FastAPI + Swagger UI + Streamlit UI; NVIDIA NIM and HuggingFace APIs |
| 7 | LLMOps + ≥5 metrics | `GET /metrics` serves M1–M6 live; M7 comes from the offline harness |
| 8 | Fine-tune on a domain dataset | DistilBERT on `cnamuangtoun/resume-job-description-fit` (6.2k rows) |

### The six sub-tasks

| # | Sub-task | Category | Model | Endpoint |
|---|---|---|---|---|
| 1 | Document-type image classification | CV | `microsoft/dit-base-finetuned-rvlcdip` | `POST /classify-document` |
| 2 | Text extraction / OCR | CV | PyMuPDF + docTR (`db_resnet50` + `crnn_vgg16_bn`) | `POST /extract-text` |
| 3 | Named entity recognition | NLP | `oksomu/resume-ner` | `POST /entities` |
| 4 | **Resume-fit classification** ★ | NLP | **fine-tuned DistilBERT** vs prompted `gpt-oss-20b` | `POST /classify-fit` |
| 5 | Extractive question answering | NLP | `deepset/roberta-base-squad2` | `POST /ask` |
| 6 | Candidate brief + interview questions | NLP | `gpt-oss-20b` → `flan-t5-base` fallback | `POST /candidate-brief` |

★ is the fine-tuning target for requirement 8.

**Why the pipeline hangs together:** a recruiter uploads a document → CV confirms it *is* a
resume and reads it → NER pulls structured facts → the fine-tuned classifier scores fit
against the JD → QA answers follow-up questions with literal resume spans → the LLM writes
the screening brief. Every sub-task feeds the next; none is decorative.

---

## 2. Design decisions worth defending in the viva

**Why a fine-tuned SLM when gpt-oss-20b could just be prompted?**
That is the experiment, not an oversight. `POST /compare-fit-models` runs both arms on the
same input and `scripts/evaluate.py` scores them on the same held-out test split. The report
presents the numbers rather than an opinion. The fine-tuned model is also ~66M params and
runs on CPU in milliseconds with no API dependency, which matters for a screening tool that
processes thousands of resumes.

**Why the dataset is class-weighted.** Labels are ~50/25/25 (No Fit / Potential / Good). An
unweighted model collapses onto "No Fit" and reports 50% accuracy having learned nothing, so
the loss is class-weighted and the headline metric is **macro-F1**, not accuracy.

**Why OCR is a fallback, not the default.** A PDF with a text layer is extracted losslessly by
PyMuPDF; running OCR over it would inject recognition errors for no benefit. docTR runs on
scans and photographs — which is exactly where the CV sub-task earns its place.

**Why docTR rather than Tesseract.** Requirement 4 asks us to *identify the models*.
Tesseract is a system binary with no model to name; docTR is two learned networks —
`db_resnet50` finds every word box, `crnn_vgg16_bn` reads it — installed by pip, listed in
the model registry, and citable in the report. It also removes the one system dependency
that made setup fail on a clean machine.

**Why QA is extractive, not generative.** Every answer is a literal span of the resume with
character offsets, so the model cannot invent a qualification the candidate does not have.

**Why there is a fallback path at all.** If `NVIDIA_API_KEY` is missing or the network is
down mid-demo, generation degrades to a local `flan-t5-base` running the *same instruction*,
and the response is stamped `degraded: true`. Availability without silently changing the
feature — the old version fell back to a summariser and quietly dropped the interview
questions.

**Fairness.** Name, age, gender, nationality and photograph are excluded from scoring and
from the generation prompt. Every response carries a decision-support disclaimer. No raw
resume text is written to the request log — only counts, latencies and model
metadata.

---

## 3. Setup

**Python 3.12** — not 3.13/3.14. The `transformers` 4.x line and its torch
dependency have no wheels for 3.13+, and Colab is on 3.12 too, so the
fine-tuned model moves between them without surprises.

```bash
py -3.12 -m venv venv           # Windows;  python3.12 -m venv venv on Unix
venv\Scripts\activate           # source venv/bin/activate on Unix
pip install -r requirements.txt
```

No system packages needed — OCR is docTR, installed by pip. (On a slim Linux
container, opencv needs `libgl1 libglib2.0-0`; the Dockerfile installs them.)

**API key** — free from https://build.nvidia.com/models :
```bash
cp .env.example .env      # then paste your nvapi-... key into it
```
Without a key everything still runs; generation degrades to the local model and
`/health` reports it.

---

## 4. Run

```bash
uvicorn app.main:app --reload --port 8000      # terminal 1
streamlit run streamlit_app.py                 # terminal 2
```

- Swagger UI → http://127.0.0.1:8000/docs — the single best report screenshot
- Demo UI → http://localhost:8501 — walk tabs ① → ⑦ in order for the viva

Set `WARMUP_MODELS=1` before the demo so the ~30s of model loading happens at
startup instead of on your first request in front of an examiner.

Verify the whole pipeline against the real models at any time:
```bash
python scripts/smoke_test.py     # every sub-task; SKIPs what is not configured
python tests/test_logic.py       # parsing/chunking/percentile self-checks
```

Containerised:
```bash
docker compose up --build      # API on :8000, UI on :8501
```

---

## 5. Fine-tuning (requirement 8)

**On Colab — do it this way.** Open [`finetune_colab.ipynb`](finetune_colab.ipynb),
set *Runtime → T4 GPU*, Run all. It writes the training script onto the VM,
trains on the full 6,241 rows, prints the before/after table and produces
`finetuned.zip`. Unzip into the project so `models/finetuned-fit-classifier/`
exists. **~8 min.**

The notebook uses no `google.colab` widgets, so it works identically in the
browser and through the VS Code Colab extension — `files.upload()` and
`files.download()` are browser-only and fail silently in the extension. The
training script is embedded via `%%writefile`, generated from the real file:

```bash
python scripts/build_colab_notebook.py    # re-run after editing finetune_classifier.py
```

Inference is CPU-only from then on — the GPU is needed for training, not serving.

**Locally on CPU — measured, not estimated: ~8 hours.** DistilBERT trains at
~0.75 samples/sec on a laptop CPU and each eval pass over the test split costs
13 minutes. If you must, trim it:
```bash
python scripts/finetune_classifier.py \
    --epochs 2 --max-train-rows 2500 --max-length 192 --eval-rows 500
```
That lands near 1.5 h at a real cost in macro-F1. Colab is the better trade.

The script writes `models/finetuned-fit-classifier/training_report.json` with the
before/after metrics and the per-epoch curve — that JSON is your training-evidence
screenshot.

---

## 6. Evaluation — the fine-tuned vs prompted head-to-head

```bash
python scripts/evaluate.py --limit 100
```
Stratified 100-row sample from the held-out test split, both arms, identical inputs.
Writes `logs/eval_report.json`: accuracy, macro-F1, per-class F1, confusion matrix,
mean latency and token cost for each arm, plus a verdict block.

That file is then served as **M7** on `GET /metrics`.

---

## 7. LLMOps (requirement 7)

Live from the request log (`GET /metrics`):

| | Metric | Detail |
|---|---|---|
| M1 | Latency | p50 / p95 / p99, overall and per endpoint |
| M2 | Throughput | requests/min, volume per endpoint |
| M3 | Reliability | success rate, error rate, errors by type and endpoint |
| M4 | Token usage | prompt / completion / total + estimated cost |
| M5 | Degradation rate | share of LLM calls that fell back to local |
| M6 | Model confidence | mean confidence per endpoint |
| M7 | Quality (offline) | macro-F1 fine-tuned vs prompted, from `scripts/evaluate.py` |

Beyond metrics, the LLMOps practices implemented are:

- **Model registry** — `GET /model-registry`; every response also stamps the model that produced it
- **Prompt versioning** — prompts are versioned artifacts (`llm.PROMPT_VERSIONS`), stamped into responses
- **Graceful degradation** — LLM failure falls back locally and is flagged, never silent
- **Structured request logging** — one JSONL record per call, no PII
- **Model caching** — every model loaded once per process via `lru_cache`
- **Health/readiness** — `GET /health` reports LLM backend and whether the fine-tuned model is loaded
- **Containerised deploy** — Dockerfile + compose, config via environment

---

## 8. Report screenshot checklist

1. `GET /docs` — full endpoint list grouped by sub-task and category
2. `GET /model-registry` — proves requirement 4
3. Streamlit tab ① — DiT document-type bar chart + extracted OCR text
4. Streamlit tab ② — grouped entities
5. **`training_report.json`** — before/after macro-F1 (requirement 8 evidence)
6. **`POST /compare-fit-models`** — fine-tuned vs prompted on one resume, side by side
7. `logs/eval_report.json` verdict block — the 100-row head-to-head
8. Streamlit tab ⑤ — generated brief with strengths, gaps and interview questions
9. Streamlit tab ⑥ — full `/screen-candidate` JSON
10. Streamlit tab ⑦ — the metrics dashboard (requirement 7 evidence)

---

## 9. Known caveats (raise these yourself in the viva — they read as rigour)

- **DiT is layout-sensitive.** RVL-CDIP's `resume` class was learned from
  scanned business documents. A plainly-formatted text render is classified as
  `email` with high confidence. Demo it with a properly formatted resume PDF,
  and note that `/ingest-resume` only *warns* on a mismatch rather than
  blocking — a classifier this brittle must not be a hard gate.
- **gpt-oss-20b is a reasoning model.** It spends completion tokens on hidden
  reasoning before emitting content, so a tight `max_tokens` returns an *empty
  string* with `finish_reason="length"`. `llm.chat` adds a fixed
  `REASONING_TOKEN_HEADROOM` on top of every caller's budget and flags
  `truncated` when it still runs out.
- **First run downloads ~2 GB** of weights (DiT, BERT-NER, RoBERTa-QA, docTR
  detection + recognition, flan-t5). Do this before the demo, not during it.
- **Without `NVIDIA_API_KEY` the prompted arm is meaningless.** It degrades to
  `flan-t5-base`, which cannot do the task; `scripts/evaluate.py` deliberately
  refuses to score that arm rather than publishing a fake comparison.
- **Fit classification truncates at 512 tokens** — DistilBERT's architectural
  ceiling. The median resume+JD pair is ~1,460 tokens, so the model reads about
  a third of each pair. JD-guided selection was built and trained as an
  alternative to head truncation; see §6 of the report for the ablation.

---

## 10. Structure

```
recruit_ai_app/
├── app/
│   ├── config.py       settings + model registry
│   ├── llm.py          NVIDIA NIM client, prompt versions, fallback
│   ├── pipeline.py     the 6 sub-tasks + orchestration
│   ├── metrics.py      LLMOps M1–M7
│   └── main.py         FastAPI routes, upload validation, request tracking
├── scripts/
│   ├── finetune_classifier.py    requirement 8
│   └── evaluate.py               fine-tuned vs prompted head-to-head
│   ├── smoke_test.py             every sub-task against the real models
│   └── build_colab_notebook.py   regenerates finetune_colab.ipynb
├── finetune_colab.ipynb  GPU training, browser- and VS-Code-compatible
├── tests/test_logic.py logic self-checks (no framework: just run it)
├── data/               sample_resume.txt, sample_jd.txt
├── streamlit_app.py    demo UI (pure API client)
├── Dockerfile · docker-compose.yml
└── requirements.txt
```
