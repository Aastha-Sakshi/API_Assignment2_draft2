"""
AI Recruitment Assistant — core pipeline
Domain: HR / Recruitment
Categories: Computer Vision (CV) + Natural Language Processing (NLP)

Sub-tasks (6):
  1. Document-type image classification  (CV)   microsoft/dit-base-finetuned-rvlcdip
  2. Text extraction / OCR               (CV)   PyMuPDF + docTR (db_resnet50 + crnn_vgg16_bn)
  3. Named Entity Recognition            (NLP)  dslim/bert-base-NER
  4. Resume-fit classification           (NLP)  DistilBERT (FINE-TUNED) vs prompted gpt-oss-20b
  5. Extractive Question Answering       (NLP)  deepset/roberta-base-squad2
  6. Candidate brief + interview questions (NLP) gpt-oss-20b -> flan-t5-base fallback

Unified objective: screen one candidate against one job description.

Models are lazy-loaded and cached (`lru_cache`) so the API starts instantly
and each model is loaded from disk at most once per process.
"""

import io
import re
import time
import logging
from functools import lru_cache
from typing import Dict, List, Optional

from app import config, llm

logger = logging.getLogger("pipeline")

# RVL-CDIP has 16 document classes. These are the ones a recruiter upload
# could legitimately be; anything else is flagged as probably-not-a-resume.
RESUME_LIKE_DOC_CLASSES = {"resume", "letter", "form", "memo"}


# ---------------------------------------------------------------------------
# Sub-task 1: document-type image classification (Computer Vision)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _doc_type_pipeline():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("image-classification", model=config.DOC_TYPE_MODEL)


def classify_document_type(image_bytes: bytes, top_k: int = 3) -> Dict:
    """
    Is the uploaded page actually a resume? Runs the DiT document-image
    classifier before any text work, so the pipeline fails fast and
    explainably on a payslip or a scanned ID card.
    """
    from PIL import Image

    start = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preds = _doc_type_pipeline()(image, top_k=top_k)

    top = preds[0]
    return {
        "predicted_type": top["label"],
        "confidence": round(float(top["score"]), 4),
        "is_resume_like": top["label"].lower() in RESUME_LIKE_DOC_CLASSES,
        "top_k": [{"label": p["label"], "score": round(float(p["score"]), 4)} for p in preds],
        "model": config.MODEL_REGISTRY["doc_type"],
        "latency_sec": round(time.perf_counter() - start, 3),
    }


# ---------------------------------------------------------------------------
# Sub-task 2: text extraction / OCR (Computer Vision)
# ---------------------------------------------------------------------------
def render_pdf_page(pdf_bytes: bytes, page_index: int = 0, dpi: int = 150) -> bytes:
    """Rasterise one PDF page to PNG bytes. Used to feed the CV classifier."""
    import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_index >= doc.page_count:
            raise ValueError(f"PDF has {doc.page_count} page(s); page {page_index} requested")
        pixmap = doc[page_index].get_pixmap(dpi=dpi)
        return pixmap.tobytes("png")


@lru_cache(maxsize=1)
def _ocr_predictor():
    """
    docTR OCR: a text-detection CNN locates every word box, then a recognition
    CNN+RNN reads each box. Both are learned models with published weights,
    installed by pip — unlike a system OCR binary, they are nameable artifacts
    that belong in the model registry.
    """
    from doctr.models import ocr_predictor

    return ocr_predictor(
        det_arch=config.DOCTR_DET_ARCH,
        reco_arch=config.DOCTR_RECO_ARCH,
        pretrained=True,
    )


def ocr_image(image_bytes: bytes) -> str:
    from doctr.io import DocumentFile

    return _ocr_predictor()(DocumentFile.from_images([image_bytes])).render().strip()


def ocr_pdf(pdf_bytes: bytes, max_pages: int = 5) -> str:
    """docTR rasterises the PDF itself, so scanned PDFs need no separate step."""
    from doctr.io import DocumentFile

    pages = DocumentFile.from_pdf(pdf_bytes)[:max_pages]
    return _ocr_predictor()(pages).render().strip()


def extract_text(file_bytes: bytes, suffix: str) -> Dict:
    """
    Digital text first, OCR only when needed.

    A text-layer PDF is extracted losslessly by PyMuPDF; OCR on such a file
    would inject recognition errors for no benefit. OCR is the path for scans
    and photographs — which is where the CV sub-task earns its place.
    """
    import fitz

    start = time.perf_counter()
    suffix = suffix.lower()

    # The route allowlists this already; the guard is for direct callers, where
    # anything unrecognised would otherwise fall through to the image branch and
    # fail deep inside docTR with "unable to read file".
    if suffix not in config.ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(
            f"Unsupported suffix {suffix!r}; expected one of "
            f"{sorted(config.ALLOWED_UPLOAD_SUFFIXES)} (a file suffix, not a MIME type)"
        )

    if suffix == ".txt":
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        method = "plain-text"

    elif suffix == ".pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
        text = "\n".join(p for p in pages if p.strip()).strip()
        method = "pymupdf-text-layer"

        if len(text) < 100:  # scanned PDF: no usable text layer
            text = ocr_pdf(file_bytes)
            method = "doctr-ocr"
    else:
        text = ocr_image(file_bytes)
        method = "doctr-ocr"

    if len(text) < 30:
        raise ValueError(
            "Extracted too little readable text. The document may be blank, "
            "encrypted, or a low-quality scan."
        )

    return {
        "text": clean_text(text),
        "char_count": len(text),
        "extraction_method": method,
        "model": config.MODEL_REGISTRY["ocr" if method == "doctr-ocr" else "text_layer"],
        "latency_sec": round(time.perf_counter() - start, 3),
    }


def clean_text(text: str) -> str:
    """Normalise whitespace but keep line breaks — sections and bullets need them."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sub-task 3: Named Entity Recognition (NLP)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ner_pipeline():
    from transformers import AutoTokenizer, pipeline as hf_pipeline

    tokenizer = AutoTokenizer.from_pretrained(config.NER_MODEL)

    # oksomu/resume-ner ships a BERT-style tokenizer config alongside a
    # DistilBERT model. The tokenizer emits token_type_ids, which
    # DistilBertForTokenClassification.forward() does not accept, so the
    # pipeline dies with a TypeError on the first call. Keyed off the model
    # architecture rather than the repo name, which carries no hint of it.
    from transformers import AutoConfig

    if AutoConfig.from_pretrained(config.NER_MODEL).model_type == "distilbert":
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    return hf_pipeline(
        "ner",
        model=config.NER_MODEL,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def _chunk(text: str, max_chars: int = 1200) -> List[str]:
    """Chunk on line boundaries so entities are not split mid-token."""
    chunks, current, size = [], [], 0
    for line in text.split("\n"):
        if current and size + len(line) > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def extract_entities(
    resume_text: str, min_confidence: float = 0.60, _include_personal_spans: bool = False
) -> Dict:
    """
    Chunked so long resumes are fully processed. The previous version
    truncated at 2000 chars, silently dropping the entire work history of any
    resume longer than about one page.

    Screening entities (skills, titles, degrees, certifications) are returned
    separately from personal identifiers. The model detects both; only the
    first group is allowed to influence any downstream judgement. Detecting
    PII explicitly is what makes "we excluded it" a verifiable claim rather
    than an assertion.

    `_include_personal_spans` is internal and deliberately not exposed by any
    route: the detected identifier *strings* are needed by redact_pii, but
    returning them in an API response would publish the very data the rest of
    this function exists to withhold. Callers get counts and types only.
    """
    start = time.perf_counter()
    ner = _ner_pipeline()
    chunks = _chunk(resume_text)

    seen, entities, personal = set(), [], []
    for chunk in chunks:
        for ent in ner(chunk):
            score = float(ent["score"])
            # Spans routinely include the delimiter that followed them
            # ("Python,", "AWS."). Trailing punctuation is not part of the
            # entity, and stripping it before dedup collapses "Python" and
            # "Python," into one result instead of two.
            word = ent["word"].strip().strip(",.;:|/-–—•() ").strip()
            label = ent["entity_group"].upper()
            key = (label, word.lower())
            if score < min_confidence or key in seen or not word:
                continue
            seen.add(key)
            record = {"entity": label, "text": word, "score": round(score, 3)}
            (personal if label in config.PII_ENTITY_TYPES else entities).append(record)

    grouped: Dict[str, List[str]] = {}
    for ent in entities:
        grouped.setdefault(ent["entity"], []).append(ent["text"])

    return {
        "entities": entities,
        "grouped": grouped,
        "count": len(entities),
        "personal_identifiers_excluded": {
            "count": len(personal),
            "types": sorted({e["entity"] for e in personal}),
            "note": "Detected and withheld from scoring and generation.",
        },
        **({"_personal_spans": personal} if _include_personal_spans else {}),
        "chunks_processed": len(chunks),
        "model": config.MODEL_REGISTRY["ner"],
        "latency_sec": round(time.perf_counter() - start, 3),
    }


# ---------------------------------------------------------------------------
# Sub-task 4: resume-fit classification (NLP) — the fine-tuning target
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _finetuned_model():
    """Loaded once per process. Reloading per request cost ~2s of pure waste."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = str(config.FINETUNED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def finetuned_available() -> bool:
    return (config.FINETUNED_DIR / "config.json").exists()


@lru_cache(maxsize=1)
def finetuned_selection_strategy() -> str:
    """
    Which input-preparation strategy this checkpoint was TRAINED with.

    Read from the model directory, not from config: a model trained on
    head-truncated text and served JD-selected text (or vice versa) sees an
    input distribution it never learned, and degrades with no error anywhere.
    Shipping the strategy alongside the weights makes the two impossible to
    desynchronise. Older checkpoints predate the file and were head-truncated.
    """
    marker = config.FINETUNED_DIR / "selection_strategy.txt"
    if not marker.exists():
        return "head"
    value = marker.read_text(encoding="utf-8").strip()
    return value if value in ("head", "jd_guided") else "head"


def classify_fit_finetuned(resume_text: str, job_description: str) -> Dict:
    """
    JD-guided selection runs here and, identically, during fine-tuning. If the
    two ever diverge the model is served a different input distribution than
    it was trained on, which degrades quietly rather than failing loudly.
    """
    import torch

    start = time.perf_counter()
    tokenizer, model = _finetuned_model()
    strategy = finetuned_selection_strategy()

    if strategy == "jd_guided":
        from app import selection

        jd_used, resume_used = selection.prepare_pair(
            resume_text,
            job_description,
            max_tokens=config.SELECTION_MAX_TOKENS,
            embedder=config.SELECTION_EMBEDDER or None,
        )
    else:
        jd_used, resume_used = job_description, resume_text

    inputs = tokenizer(
        jd_used,
        resume_used,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()

    # Read labels off the model config — never assume the training label order
    # matched a hard-coded list in application code.
    id2label = model.config.id2label
    scores = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    label = id2label[int(torch.argmax(logits))]

    return {
        "label": label,
        "confidence": scores[label],
        "scores": scores,
        "method": "finetuned",
        "model": config.MODEL_REGISTRY["fit_finetuned"],
        "input_preparation": {
            "strategy": strategy,          # read from the checkpoint, not config
            "embedder": (config.SELECTION_EMBEDDER or "tf-idf") if strategy == "jd_guided" else None,
            "budget_tokens": config.SELECTION_MAX_TOKENS,
            "resume_chars_in": len(resume_text),
            "resume_chars_used": len(resume_used),
        },
        "latency_sec": round(time.perf_counter() - start, 3),
    }


FIT_PROMPT = """You are a recruitment screening assistant.
Classify how well the candidate's resume fits the job description.

Answer with EXACTLY one of these labels and nothing else:
No Fit
Potential Fit
Good Fit

--- JOB DESCRIPTION ---
{jd}

--- RESUME ---
{resume}

Label:"""


def classify_fit_prompted(
    resume_text: str,
    job_description: str,
    match_slm_input: bool = False,
) -> Dict:
    """
    Zero-shot baseline — the control arm for the fine-tuning experiment.

    `match_slm_input=True` feeds the LLM exactly the 512 tokens the fine-tuned
    model sees, instead of the whole document. Without that arm the comparison
    confounds two variables: the LLM reads ~3x more evidence AND has 300x more
    parameters, so a win cannot be attributed to either. Running both LLM
    conditions separates them:

        LLM(full) - LLM(matched)   = the context effect
        LLM(matched) - DistilBERT  = the capability effect
    """
    if match_slm_input:
        # "Matched" has to mean matched to the SHIPPED checkpoint, not to
        # whichever preparation is most sophisticated. The ablation picked head
        # truncation, so applying JD-guided selection here would feed Arm C a
        # different 512 tokens from Arm A and quietly reintroduce the confound
        # this arm exists to remove. Read the strategy, do not assume it.
        strategy = finetuned_selection_strategy() if finetuned_available() else "head"

        if strategy == "jd_guided":
            from app import selection

            jd_used, resume_used = selection.prepare_pair(
                resume_text,
                job_description,
                max_tokens=config.SELECTION_MAX_TOKENS,
                embedder=config.SELECTION_EMBEDDER or None,
            )
        else:
            # Same budget, same split, same order the classifier's tokenizer
            # would truncate to -- expressed in characters.
            from app.selection import CHARS_PER_TOKEN

            per_side = config.SELECTION_MAX_TOKENS // 2 * CHARS_PER_TOKEN
            jd_used, resume_used = job_description[:per_side], resume_text[:per_side]
    else:
        jd_used, resume_used = job_description[:3000], resume_text[:5000]

    result = llm.chat(
        FIT_PROMPT.format(jd=jd_used, resume=resume_used),
        max_tokens=32,  # llm.chat adds reasoning headroom on top of this
        temperature=0.0,
    )

    raw = result["text"]
    label = _parse_fit_label(raw)

    return {
        "label": label,
        "confidence": None,  # a prompted LLM gives no calibrated probability
        "raw_output": raw,
        "parse_failed": label is None,
        "truncated": result.get("truncated", False),
        "method": "prompted_matched_input" if match_slm_input else "prompted",
        "input_chars": len(jd_used) + len(resume_used),
        "model": {**config.MODEL_REGISTRY["fit_prompted"], "prompt_version": llm.PROMPT_VERSIONS["fit_classify"]},
        "backend": result["backend"],
        "degraded": result["degraded"],
        "total_tokens": result["total_tokens"],
        "latency_sec": result["latency_sec"],
    }


def _parse_fit_label(raw: str) -> Optional[str]:
    """
    Free-text output has to be coerced back to the label set. Checked
    longest-first so 'No Fit' cannot swallow a match intended for a longer
    label, and the whole string is searched because small models prepend
    filler before the answer.
    """
    lowered = raw.lower()
    for label in sorted(config.FIT_LABELS, key=len, reverse=True):
        if label.lower() in lowered:
            return label
    return None


def classify_fit(resume_text: str, job_description: str, method: str = "auto") -> Dict:
    """
    method: 'finetuned' | 'prompted' | 'auto'
    'auto' prefers the fine-tuned model and falls back to the prompted LLM
    when it has not been trained yet.
    """
    if method == "prompted":
        return classify_fit_prompted(resume_text, job_description)

    if method == "finetuned":
        if not finetuned_available():
            raise FileNotFoundError(
                f"No fine-tuned model at {config.FINETUNED_DIR}. "
                "Run scripts/finetune_classifier.py (or the Colab notebook) first."
            )
        return classify_fit_finetuned(resume_text, job_description)

    if finetuned_available():
        return classify_fit_finetuned(resume_text, job_description)
    return classify_fit_prompted(resume_text, job_description)


# ---------------------------------------------------------------------------
# Sub-task 5: extractive Question Answering (NLP)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _qa_pipeline():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("question-answering", model=config.QA_MODEL)


# Abstention threshold on the best real span. Measured on a full one-page
# resume: answerable questions scored 0.013-0.208, unanswerable ones 0.0001-0.0006
# (one outlier at 0.156 -- see the report's limitations).
#
# Why a threshold rather than the pipeline's own `handle_impossible_answer`
# verdict: on a 5.3k-char resume the context is split into ~10 stride windows,
# each softmaxed independently. The null option is a single token pair (CLS,CLS)
# that concentrates probability mass, whereas a real answer's mass is spread over
# many candidate start/end pairs in its window. transformers keeps the MINIMUM
# null across windows (question_answering.py:141) -- and here even that minimum
# was 0.735, while the correct span scored 0.0135 in the one window containing
# it. Comparing those two numbers directly is not comparing like with like, so
# the null always wins and the endpoint abstains on every question about a real
# resume. The null score is still reported, just not given the casting vote.
QA_MIN_SPAN_SCORE = 0.005


def answer_question(resume_text: str, question: str) -> Dict:
    start = time.perf_counter()
    results = _qa_pipeline()(
        question=question,
        context=resume_text,
        max_answer_len=80,
        handle_impossible_answer=True,
        top_k=5,
    )
    if isinstance(results, dict):
        results = [results]

    spans = [r for r in results if r["answer"].strip()]
    null_score = max((float(r["score"]) for r in results if not r["answer"].strip()), default=0.0)

    best = max(spans, key=lambda r: r["score"], default=None)
    confident = best is not None and float(best["score"]) >= QA_MIN_SPAN_SCORE

    result = best if confident else {"answer": "", "score": null_score, "start": 0, "end": 0}
    answer = result["answer"].strip()
    return {
        "answer": answer or "Not stated in this resume.",
        "confidence": round(float(result["score"]), 4),
        "abstention_score": round(null_score, 4),
        "start_char": result["start"],
        "end_char": result["end"],
        "grounded": bool(answer),  # extractive: the answer is a resume span, not invented
        "model": config.MODEL_REGISTRY["qa"],
        "latency_sec": round(time.perf_counter() - start, 3),
    }


# ---------------------------------------------------------------------------
# Sub-task 6: candidate brief + interview questions (NLP)
# ---------------------------------------------------------------------------
BRIEF_PROMPT = """You are a recruitment assistant helping a hiring manager prepare for a screening call.

Write, in plain text:
SUMMARY: 3-4 sentences on this candidate's profile relative to the role.
STRENGTHS: 3 bullet points, each citing concrete evidence from the resume.
GAPS: 2 bullet points on what the resume does not evidence for this role.
QUESTIONS: 3 interview questions that probe the gaps above.

Base every statement only on the resume text. Do not invent employers, dates,
or qualifications. Personal identifiers have been replaced with placeholders
such as [NAME]; do not speculate about what they contained.

--- JOB DESCRIPTION ---
{jd}

--- RESUME ---
{resume}
"""


# Structurally-identifiable PII, matched directly against the document.
#
# The NER layer alone is not sufficient and measurement showed why: the token
# classifier returns spans reassembled from wordpieces, so a real email came
# back as "abc99 @ xyz. com" and a phone as "91 - 9876543210". Neither string
# occurs verbatim in the resume, so a literal replace silently matched nothing
# and left both in the prompt while the count still reported success. Anything
# with a reliable surface form is therefore matched by pattern, not by model.
_PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("URL", re.compile(r"https?://\S+|(?:www\.|linkedin\.com/|github\.com/)\S+", re.I)),
]

# Phone numbers need a digit-count check, not just a shape. A pattern loose
# enough to catch "+91-9876543210", "(022) 555 0143" and "9876543210" also
# matches graduation years, CGPAs and "processed 500 000 records" -- and
# over-redaction quietly degrades the brief, which is a harder failure to spot
# than under-redaction. Requiring >= 10 digits keeps real numbers intact.
_PHONE_CANDIDATE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,5})(?:[\s.-]?\d{2,5}){1,4}")
_PHONE_MIN_DIGITS = 10


def redact_pii(resume_text: str, personal: List[Dict]) -> tuple[str, int]:
    """
    Replace personal identifiers with type placeholders before the text leaves
    this machine for a third-party API.

    Instructing the model to "ignore the name" is a request, not a control: the
    name still travels and still conditions the output. Substituting the span is
    the only version of that claim which is verifiable.

    Two layers, because neither alone is enough: regex for identifiers with a
    dependable shape (email, phone, URL), NER for the ones that have none (names,
    places). Still bounded by recall on the second layer — an unusual name the
    model misses is not redacted, so this reduces exposure rather than
    eliminating it.
    """
    redacted, count = resume_text, 0

    for label, pattern in _PII_PATTERNS:
        redacted, hits = pattern.subn(f"[{label}]", redacted)
        count += hits

    def _phone(match: "re.Match") -> str:
        nonlocal count
        if sum(c.isdigit() for c in match.group()) < _PHONE_MIN_DIGITS:
            return match.group()
        count += 1
        return "[PHONE]"

    redacted = _PHONE_CANDIDATE.sub(_phone, redacted)

    # Longest span first, so "Jane Doe" is not left as "[NAME] Doe" by an
    # earlier match on "Jane". Subword fragments ("##bad") are never literal.
    for record in sorted(personal, key=lambda r: len(r["text"]), reverse=True):
        surface = record["text"]
        if not surface or surface.startswith("##") or surface not in redacted:
            continue
        count += redacted.count(surface)
        redacted = redacted.replace(surface, f"[{record['entity']}]")

    return redacted, count


def generate_candidate_brief(
    resume_text: str, job_description: str = "", personal: Optional[List[Dict]] = None
) -> Dict:
    """
    The old version fell back to a summarisation model, which silently dropped
    the interview questions. The fallback here runs the same instruction, so a
    degraded response is a weaker brief — not a different feature.

    `personal` lets a caller that has already run NER pass the detected
    identifiers in. screen_candidate has them, and re-deriving them here cost a
    second full pass over the resume for an identical result.
    """
    if personal is None:
        personal = extract_entities(resume_text, _include_personal_spans=True)["_personal_spans"]
    safe_text, redacted_count = redact_pii(resume_text, personal)

    result = llm.chat(
        BRIEF_PROMPT.format(jd=job_description[:3000] or "(not supplied)", resume=safe_text[:6000]),
        max_tokens=600,
        temperature=0.3,
    )
    return {
        "brief": result["text"],
        "pii_redacted_spans": redacted_count,
        "model": {
            **config.MODEL_REGISTRY["generation" if not result["degraded"] else "generation_fallback"],
            "prompt_version": llm.PROMPT_VERSIONS["candidate_brief"],
        },
        "backend": result["backend"],
        "degraded": result["degraded"],
        "degraded_reason": result.get("degraded_reason"),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "latency_sec": result["latency_sec"],
    }


# ---------------------------------------------------------------------------
# End-to-end orchestration — the "cohesive, unified objective"
# ---------------------------------------------------------------------------
def screen_candidate(
    resume_text: str,
    job_description: str,
    method: str = "auto",
    include_brief: bool = True,
) -> Dict:
    """Chains NER -> fit classification -> brief into one screening result."""
    start = time.perf_counter()

    ner = extract_entities(resume_text, _include_personal_spans=True)
    fit = classify_fit(resume_text, job_description, method=method)
    # Hand the already-detected identifiers to the brief rather than making it
    # run NER over the same resume a second time.
    brief = (
        generate_candidate_brief(resume_text, job_description, personal=ner["_personal_spans"])
        if include_brief else None
    )
    ner = {k: v for k, v in ner.items() if k != "_personal_spans"}

    return {
        "fit": fit,
        "entities": ner,
        "brief": brief,
        "disclaimer": (
            "Decision-support output only. This must not be the sole basis for an "
            "employment decision. Personal characteristics are excluded from scoring."
        ),
        "app_version": config.APP_VERSION,
        "pipeline_latency_sec": round(time.perf_counter() - start, 3),
    }
