"""
JD-guided extractive selection.

The problem this solves, measured on the training set:

    JD tokens       median  403
    resume tokens   median 1058
    pair median           1461 tokens
    DistilBERT ceiling      512   -> only ~34% of the pair survives

Default `longest_first` truncation keeps the *first* 34%, which on a resume is
contact details, a summary blurb and the top of the most recent job. The
requirement-bearing evidence further down is discarded.

This module keeps the *most JD-relevant* 34% instead. Same token budget,
different content. It is a preprocessing change, not an architecture change,
so it costs nothing at serve time beyond one embedding pass.

The approach is the one argued for in resume_jd_matcher_build_guide.md §5 and
§16 — compare requirements against relevant resume evidence rather than whole
documents — applied where it belongs: to what the classifier gets to read.

CRITICAL: this must run identically during fine-tuning and at inference. A
model trained on selected text and served truncated text sees a different
distribution and will silently underperform. Both paths import this module;
the Colab notebook embeds this exact file.
"""

import re
from functools import lru_cache
from typing import List, Optional, Sequence

# ~4 characters per token for English prose. Used only to convert a token
# budget into a character budget for the greedy fill; the tokenizer remains
# the final authority via its own truncation.
CHARS_PER_TOKEN = 4

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# Lines that are pure contact/PII noise. Dropping them before selection frees
# budget AND keeps protected characteristics out of the scored text, which is
# a fairness requirement, not just an optimisation.
_PII_LINE = re.compile(
    r"^\s*(?:"
    r"[\w.+-]+@[\w-]+\.[\w.]+"                      # bare email line
    r"|(?:\+?\d[\d\s().-]{7,}\d)"                   # bare phone line
    r"|(?:https?://|www\.)\S+"                      # bare URL line
    r"|(?:address|d\.?o\.?b\.?|date of birth|gender|nationality|marital status)\s*[:\-]"
    r")\s*$",
    re.IGNORECASE,
)


def split_units(text: str, min_chars: int = 25) -> List[str]:
    """
    Split a document into selectable units.

    Resumes are line-oriented (bullets, headings), not paragraph-oriented, so
    lines are the natural unit. Very long lines are split on sentence
    boundaries so a single wall-of-text paragraph cannot monopolise the budget.
    """
    units: List[str] = []

    for line in text.split("\n"):
        line = line.strip(" \t•●▪-–—*")
        if len(line) < min_chars or _PII_LINE.match(line):
            continue
        if len(line) <= 400:
            units.append(line)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if len(sentence) >= min_chars:
                units.append(sentence)

    return units


@lru_cache(maxsize=2)
def _embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _semantic_scores(query: str, units: Sequence[str], model_name: str):
    import numpy as np

    model = _embedder(model_name)
    vectors = model.encode(
        [query, *units],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors[1:] @ vectors[0])


def _lexical_scores(query: str, units: Sequence[str]):
    """
    TF-IDF fallback for when no embedder is available (offline, or the Colab
    training job choosing not to pay for one). Weaker: it cannot match
    "REST API design" to "built Flask endpoints", which is exactly the kind of
    paraphrase this task is full of.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(
        [query, *units]
    )
    query_vector = matrix[0]
    unit_vectors = matrix[1:]
    scores = (unit_vectors @ query_vector.T).toarray().ravel()
    norms = np.sqrt(unit_vectors.multiply(unit_vectors).sum(axis=1)).A.ravel()
    denominator = norms * np.sqrt(query_vector.multiply(query_vector).sum())
    return np.divide(scores, denominator, out=np.zeros_like(scores), where=denominator > 0)


def select_relevant(
    text: str,
    query: str,
    max_tokens: int,
    embedder: Optional[str] = DEFAULT_EMBEDDER,
) -> str:
    """
    Return the subset of `text` most relevant to `query`, within `max_tokens`.

    Selected units are re-emitted in their **original document order**, not in
    score order: a resume read out of sequence loses the chronology that makes
    "5 years at X, then Y" interpretable, and the encoder sees position.

    Falls back to the head of the text if there is nothing to select from, so
    this can never return less information than plain truncation.
    """
    budget_chars = max_tokens * CHARS_PER_TOKEN
    units = split_units(text)

    if not units:
        return text[:budget_chars].strip()

    # Already fits: selection would only risk dropping evidence.
    if sum(len(u) + 1 for u in units) <= budget_chars:
        return "\n".join(units)

    try:
        scores = _semantic_scores(query, units, embedder) if embedder else _lexical_scores(query, units)
    except Exception:
        scores = _lexical_scores(query, units)

    ranked = sorted(range(len(units)), key=lambda i: float(scores[i]), reverse=True)

    chosen: List[int] = []
    used = 0
    for index in ranked:
        cost = len(units[index]) + 1
        if used + cost > budget_chars:
            continue          # keep scanning: a shorter unit may still fit
        chosen.append(index)
        used += cost

    if not chosen:            # every unit individually exceeds the budget
        return units[ranked[0]][:budget_chars].strip()

    return "\n".join(units[i] for i in sorted(chosen))


def prepare_pair(
    resume_text: str,
    job_description: str,
    max_tokens: int = 512,
    embedder: Optional[str] = DEFAULT_EMBEDDER,
) -> tuple[str, str]:
    """
    Fit a (job_description, resume) pair into `max_tokens` for a 512-token
    encoder. Returned in (jd, resume) order, matching the tokenizer call in
    the training script where the JD is passed first.

    The two sides are treated ASYMMETRICALLY, and that asymmetry is the whole
    point:

      * The job description is kept in document order and merely truncated.
        It defines the requirements. Selecting JD lines by their similarity to
        the resume would delete precisely the requirements the candidate fails
        to meet -- which is the evidence for "No Fit" -- and bias every
        prediction toward a match. An earlier version of this function did
        exactly that and measurably hurt evidence coverage.

      * The resume is filtered by relevance to the JD. Here the ranking is
        sound: a resume is a pile of evidence, most of it irrelevant to any
        one role, and the goal is to surface the part that bears on this role
        rather than whatever happens to appear in the first 254 tokens.
    """
    per_side = (max_tokens - 3) // 2      # [CLS] jd [SEP] resume [SEP]

    jd_units = split_units(job_description)
    jd_text = "\n".join(jd_units) if jd_units else job_description
    jd_kept = jd_text[: per_side * CHARS_PER_TOKEN].strip()

    resume_kept = select_relevant(resume_text, job_description, per_side, embedder)

    return jd_kept, resume_kept
