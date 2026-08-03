"""
AI Recruitment Assistant — FastAPI backend
CCZG506 Assignment II

    uvicorn app.main:app --reload --port 8000
    -> http://127.0.0.1:8000/docs   (Swagger UI: the main report screenshot)

Six sub-tasks across two categories, all serving one objective:
screen a candidate against a job description.
"""

import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import config, metrics, pipeline

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Optional model warm-up. Loading DiT/BERT/RoBERTa from disk costs ~30s in
    total; paying it at startup means the first request of a live demo is not
    the slow one. Off by default so `--reload` stays usable during development.
    """
    if os.getenv("WARMUP_MODELS") == "1":
        for name, loader in (
            ("doc_type", pipeline._doc_type_pipeline),
            ("ner", pipeline._ner_pipeline),
            ("qa", pipeline._qa_pipeline),
            # docTR is the most expensive cold load of the set -- two networks,
            # measured at 25-37s on the first OCR request. Omitting it left the
            # single slowest path in the demo unwarmed.
            ("ocr", pipeline._ocr_predictor),
        ):
            start = time.perf_counter()
            loader()
            logger.info("warmed %s in %.1fs", name, time.perf_counter() - start)
        if pipeline.finetuned_available():
            pipeline._finetuned_model()
            logger.info("warmed fine-tuned classifier")
    yield


app = FastAPI(
    title="AI Recruitment Assistant",
    version=config.APP_VERSION,
    lifespan=lifespan,
    description=(
        "API-driven AI project — HR/Recruitment domain.\n\n"
        "**Computer Vision:** document-type classification (DiT), OCR / text extraction.\n\n"
        "**NLP:** named entity recognition, resume-fit classification "
        "(fine-tuned DistilBERT vs prompted gpt-oss-20b), extractive QA, "
        "candidate brief generation.\n\n"
        "All sub-tasks are orchestrated by `POST /screen-candidate`."
    ),
)


# ---------------------------------------------------------------------------
# Request logging helper — one place, so no endpoint can forget to log
# ---------------------------------------------------------------------------
@contextmanager
def track(endpoint: str):
    """
    Yields a dict; anything put in it is merged into the log record.
    Failures are logged with their type before the HTTP error propagates,
    which is what makes the error-rate metric trustworthy.
    """
    extra: dict = {}
    start = time.perf_counter()
    try:
        yield extra
    except HTTPException as exc:
        metrics.log_event(endpoint, time.perf_counter() - start, False,
                          {**extra, "error": f"HTTP{exc.status_code}: {exc.detail}"})
        raise
    except Exception as exc:
        metrics.log_event(endpoint, time.perf_counter() - start, False,
                          {**extra, "error": f"{type(exc).__name__}: {exc}"})
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    else:
        metrics.log_event(endpoint, time.perf_counter() - start, True, extra)


async def read_upload(file: UploadFile) -> tuple[bytes, str]:
    """
    Trust boundary. Resumes are attacker-controlled files carrying personal
    data, so validate extension and size before anything touches a parser.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(config.ALLOWED_UPLOAD_SUFFIXES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    return data, suffix


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TextIn(BaseModel):
    text: str = Field(..., min_length=20, examples=["Senior Python engineer with 6 years..."])


class ClassifyIn(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description: str = Field(..., min_length=10)
    method: Literal["auto", "finetuned", "prompted"] = "auto"


class QAIn(BaseModel):
    resume_text: str = Field(..., min_length=20)
    question: str = Field(..., min_length=3, examples=["How many years of AWS experience?"])


class BriefIn(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description: str = ""


class ScreenIn(ClassifyIn):
    include_brief: bool = True


# ---------------------------------------------------------------------------
# Sub-task 1 — document-type classification (Computer Vision)
# ---------------------------------------------------------------------------
@app.post("/classify-document", tags=["CV 1 · Document classification"])
async def classify_document(file: UploadFile = File(...)):
    """Checks the upload really is a resume before spending compute on it."""
    with track("/classify-document") as log:
        data, suffix = await read_upload(file)
        image_bytes = pipeline.render_pdf_page(data) if suffix == ".pdf" else data
        if suffix == ".txt":
            raise HTTPException(status_code=415, detail="Document classification needs an image or PDF page")

        result = pipeline.classify_document_type(image_bytes)
        log["confidence"] = result["confidence"]
        log["predicted_type"] = result["predicted_type"]
        return result


# ---------------------------------------------------------------------------
# Sub-task 2 — text extraction / OCR (Computer Vision)
# ---------------------------------------------------------------------------
@app.post("/extract-text", tags=["CV 2 · OCR / text extraction"])
async def extract_text(file: UploadFile = File(...)):
    """Digital text layer when present; docTR OCR when the page is a scan."""
    with track("/extract-text") as log:
        data, suffix = await read_upload(file)
        try:
            result = pipeline.extract_text(data, suffix)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        log["extraction_method"] = result["extraction_method"]
        log["char_count"] = result["char_count"]
        return result


@app.post("/ingest-resume", tags=["CV 2 · OCR / text extraction"])
async def ingest_resume(file: UploadFile = File(...)):
    """Both CV sub-tasks on one upload: verify the document, then read it."""
    with track("/ingest-resume") as log:
        data, suffix = await read_upload(file)

        doc_type = None
        if suffix != ".txt":
            image_bytes = pipeline.render_pdf_page(data) if suffix == ".pdf" else data
            doc_type = pipeline.classify_document_type(image_bytes)

        try:
            extraction = pipeline.extract_text(data, suffix)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        log["extraction_method"] = extraction["extraction_method"]
        return {
            "document_type": doc_type,
            "extraction": extraction,
            "warning": (
                None if doc_type is None or doc_type["is_resume_like"]
                else f"Document looks like '{doc_type['predicted_type']}', not a resume. Results may be unreliable."
            ),
        }


# ---------------------------------------------------------------------------
# Sub-task 3 — Named Entity Recognition (NLP)
# ---------------------------------------------------------------------------
@app.post("/entities", tags=["NLP 3 · Named entity recognition"])
async def entities(payload: TextIn):
    with track("/entities") as log:
        result = pipeline.extract_entities(payload.text)
        log["entity_count"] = result["count"]
        return result


# ---------------------------------------------------------------------------
# Sub-task 4 — resume-fit classification (NLP, fine-tuned)
# ---------------------------------------------------------------------------
@app.post("/classify-fit", tags=["NLP 4 · Fit classification (fine-tuned)"])
async def classify_fit(payload: ClassifyIn):
    """
    `method=finetuned` uses the DistilBERT model trained on
    cnamuangtoun/resume-job-description-fit.
    `method=prompted` is the zero-shot gpt-oss-20b control arm.
    """
    with track("/classify-fit") as log:
        try:
            result = pipeline.classify_fit(payload.resume_text, payload.job_description, payload.method)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        log["method"] = result["method"]
        log["label"] = result["label"]
        if result.get("confidence") is not None:
            log["confidence"] = result["confidence"]
        for key in ("total_tokens", "degraded"):
            if key in result:
                log[key] = result[key]
        return result


@app.post("/compare-fit-models", tags=["NLP 4 · Fit classification (fine-tuned)"])
async def compare_fit_models(payload: ClassifyIn):
    """Runs both arms on the same input — the side-by-side report screenshot."""
    with track("/compare-fit-models"):
        finetuned = (
            pipeline.classify_fit_finetuned(payload.resume_text, payload.job_description)
            if pipeline.finetuned_available()
            else {"error": f"No fine-tuned model at {config.FINETUNED_DIR}"}
        )
        prompted = pipeline.classify_fit_prompted(payload.resume_text, payload.job_description)
        # Two missing labels are not an agreement. Only compare real verdicts.
        labels = (finetuned.get("label"), prompted.get("label"))
        return {
            "finetuned": finetuned,
            "prompted": prompted,
            "agree": all(labels) and labels[0] == labels[1],
        }


# ---------------------------------------------------------------------------
# Sub-task 5 — extractive Question Answering (NLP)
# ---------------------------------------------------------------------------
@app.post("/ask", tags=["NLP 5 · Question answering"])
async def ask(payload: QAIn):
    with track("/ask") as log:
        result = pipeline.answer_question(payload.resume_text, payload.question)
        # On abstention `confidence` carries the null-answer score, which is
        # confidence that the question is UNANSWERABLE — not confidence in an
        # answer. Logging it would push ~0.76 into M6 mean-confidence every time
        # the model correctly declined, making the metric read highest when the
        # endpoint answered least.
        log["confidence"] = result["confidence"] if result["grounded"] else None
        log["abstained"] = not result["grounded"]
        return result


# ---------------------------------------------------------------------------
# Sub-task 6 — candidate brief + interview questions (NLP)
# ---------------------------------------------------------------------------
@app.post("/candidate-brief", tags=["NLP 6 · Brief generation"])
async def candidate_brief(payload: BriefIn):
    with track("/candidate-brief") as log:
        result = pipeline.generate_candidate_brief(payload.resume_text, payload.job_description)
        log.update({
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "degraded": result["degraded"],
        })
        return result


# ---------------------------------------------------------------------------
# Orchestration — the unified objective
# ---------------------------------------------------------------------------
@app.post("/screen-candidate", tags=["0 · Full pipeline"])
async def screen_candidate(payload: ScreenIn):
    with track("/screen-candidate") as log:
        try:
            result = pipeline.screen_candidate(
                payload.resume_text, payload.job_description,
                method=payload.method, include_brief=payload.include_brief,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        log["label"] = result["fit"]["label"]
        if result["brief"]:
            # All three counts, not just the total: M4 sums prompt, completion
            # and total independently across the log, so an endpoint that logs
            # only the total makes the dashboard self-contradictory -- the
            # reported total stops equalling the reported parts.
            log.update({
                "prompt_tokens": result["brief"]["prompt_tokens"],
                "completion_tokens": result["brief"]["completion_tokens"],
                "total_tokens": result["brief"]["total_tokens"],
                "degraded": result["brief"]["degraded"],
            })
        return result


# ---------------------------------------------------------------------------
# LLMOps
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["LLMOps"])
async def metrics_dashboard():
    """The >= 5 required metrics, computed from the live request log."""
    return metrics.summarize_metrics()


@app.get("/health", tags=["LLMOps"])
async def health():
    return {
        "status": "ok",
        "app_version": config.APP_VERSION,
        "llm_backend": config.NVIDIA_MODEL if config.llm_available() else f"{config.FALLBACK_GEN_MODEL} (degraded)",
        "llm_configured": config.llm_available(),
        "finetuned_model_loaded": pipeline.finetuned_available(),
    }


@app.get("/model-registry", tags=["LLMOps"])
async def model_registry():
    """Every model this service can invoke, by sub-task and category."""
    return {"app_version": config.APP_VERSION, "models": config.MODEL_REGISTRY}
