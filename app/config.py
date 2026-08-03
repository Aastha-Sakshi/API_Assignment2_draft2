"""
Central configuration + model registry.

The MODEL_REGISTRY is not decoration: LLMOps requires that every response can
be traced back to the exact model artifact that produced it. Every endpoint
stamps its entry into the response, so a screenshot in the report is
self-documenting.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_VERSION = "2.0.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- NVIDIA NIM (OpenAI-compatible endpoint) -------------------------------
# Free API keys: https://build.nvidia.com/models
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

# --- Local model ids -------------------------------------------------------
DOC_TYPE_MODEL = os.getenv("DOC_TYPE_MODEL", "microsoft/dit-base-finetuned-rvlcdip")
# docTR runs OCR as two learned stages: text detection then text recognition.
DOCTR_DET_ARCH = os.getenv("DOCTR_DET_ARCH", "db_resnet50")
DOCTR_RECO_ARCH = os.getenv("DOCTR_RECO_ARCH", "crnn_vgg16_bn")
# Resume-domain NER. dslim/bert-base-NER (CoNLL-03) only emits PER/ORG/LOC/MISC
# — no skills, degrees, titles or certifications, i.e. nothing a recruiter
# screens on. This model's schema is SKILL/TITLE/COMPANY/DEGREE/INSTITUTION/
# CERT/FIELD/LANGUAGE/DATE plus the PII types below.
NER_MODEL = os.getenv("NER_MODEL", "oksomu/resume-ner")

# Personal identifiers. Detecting them is what makes any exclusion claim
# checkable, but be precise about what is actually enforced where:
#
#   /entities          they are counted and typed, never returned as text.
#   /candidate-brief   the spans are substituted out of the prompt by
#                      pipeline.redact_pii before it leaves this machine.
#   /classify-fit      NOT redacted. The fine-tuned model was trained on raw
#                      resume text, so redacting at inference would feed it an
#                      input distribution it never saw. The model is local and
#                      returns only a 3-way label, but the text does reach it.
#
# Bounded by NER recall throughout: an identifier the model fails to detect is
# not redacted. This reduces exposure; it does not eliminate it.
PII_ENTITY_TYPES = {"NAME", "EMAIL", "PHONE", "LOCATION"}

# gpt-oss-20b exposes OpenAI's reasoning_effort control. Measured on the brief
# prompt: "low" produced 393 completion tokens against 820 with the default and
# 937 at "medium", for output of comparable length (1,720 vs 1,803 chars) --
# roughly half the tokens billed, half the exposure to the reasoning-budget
# ceiling, and no visible quality loss on this task. Set to "" to send nothing.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")

# JD-guided extractive selection (see app/selection.py). Set to "" to fall
# back to TF-IDF, which needs no model download.
# Empty => TF-IDF. Measured on 40 pairs: TF-IDF 11 ms/pair, all-MiniLM-L6-v2
# 1764 ms/pair for no gain in JD-term recall (0.061 vs 0.051, head 0.053).
# A 1.8 s per-request cost would erase the fine-tuned model's only advantage
# over the prompted LLM, which is latency. Semantic selection stays available
# behind this switch; it is not the default.
SELECTION_EMBEDDER = os.getenv("SELECTION_EMBEDDER", "")
SELECTION_MAX_TOKENS = int(os.getenv("SELECTION_MAX_TOKENS", "512"))
QA_MODEL = os.getenv("QA_MODEL", "deepset/roberta-base-squad2")
FALLBACK_GEN_MODEL = os.getenv("FALLBACK_GEN_MODEL", "google/flan-t5-base")
FINETUNED_DIR = Path(os.getenv("FINETUNED_DIR", PROJECT_ROOT / "models" / "finetuned-fit-classifier"))

# --- Upload guardrails (resumes are sensitive personal data) ---------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".txt"}

# --- Fine-tune / eval defaults --------------------------------------------
FIT_DATASET = os.getenv("FIT_DATASET", "cnamuangtoun/resume-job-description-fit")
FIT_LABELS = ["No Fit", "Potential Fit", "Good Fit"]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
REQUEST_LOG = LOG_DIR / "requests.jsonl"


MODEL_REGISTRY = {
    "doc_type": {"task": "image-classification", "model": DOC_TYPE_MODEL, "category": "CV"},
    "ocr": {
        "task": "optical-character-recognition",
        "model": f"docTR {DOCTR_DET_ARCH} (detection) + {DOCTR_RECO_ARCH} (recognition)",
        "category": "CV",
    },
    "text_layer": {"task": "document-parsing", "model": "PyMuPDF (no ML)", "category": "CV"},
    "ner": {"task": "token-classification", "model": NER_MODEL, "category": "NLP"},
    "selection": {
        "task": "sentence-similarity",
        "model": SELECTION_EMBEDDER or "tf-idf (no model)",
        "category": "NLP",
        # The ablation (6 epochs @ 512 on 1,759 held-out rows) measured JD-guided
        # selection at macro-F1 0.3776 against head truncation's 0.4004, so the
        # shipped checkpoint is head-truncated and this component is not on the
        # serving path. Kept because it is the trained alternative arm and the
        # notebook can still select it; the registry should say which is live.
        "role": "JD-guided extractive selection — evaluated, not shipped (head truncation won)",
    },
    "fit_finetuned": {"task": "text-classification", "model": "distilbert-base-uncased (fine-tuned)", "category": "NLP"},
    "fit_prompted": {"task": "text-classification", "model": NVIDIA_MODEL, "category": "NLP"},
    "qa": {"task": "question-answering", "model": QA_MODEL, "category": "NLP"},
    "generation": {"task": "text-generation", "model": NVIDIA_MODEL, "category": "NLP"},
    "generation_fallback": {"task": "text2text-generation", "model": FALLBACK_GEN_MODEL, "category": "NLP"},
}


def llm_available() -> bool:
    return bool(NVIDIA_API_KEY)
