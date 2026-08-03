"""Ablation: head truncation vs JD-guided selection, at an identical budget.

Metric: of the job description's content vocabulary, what fraction is
evidenced *in the resume half* of what the classifier reads. This is the
question that matters -- can the model see the evidence for the requirements?
Measuring over the combined jd+resume text instead would be confounded, since
head truncation trivially retains JD terms in the JD half.

Selection is timed after a warm-up call so model load is not charged to it.
"""
import re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.selection import select_relevant, split_units  # noqa: E402

STOP = set('and the for with a an of to in on at is are be as by or from that this you we our will '
           'have has been their they it its can not but if then than each other more most all any'.split())
CHARS_PER_TOKEN = 4


def content_words(text):
    return {w for w in re.findall(r'[a-z][a-z0-9+#.\-]{2,}', text.lower()) if w not in STOP}


def main():
    from datasets import load_dataset

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rows = list(load_dataset('cnamuangtoun/resume-job-description-fit', split='train').select(range(n)))
    budget = (512 - 3) // 2                      # tokens available to the resume half
    budget_chars = budget * CHARS_PER_TOKEN

    print(f'{n} pairs | resume half budget = {budget} tokens\n', flush=True)
    print(f'{"strategy":24}{"JD-term recall":>16}{"delta":>9}{"ms/pair":>10}', flush=True)

    head_recall = []
    for r in rows:
        jd_terms = content_words(r['job_description_text'])
        head = content_words(r['resume_text'][:budget_chars])
        head_recall.append(len(jd_terms & head) / max(len(jd_terms), 1))
    base = float(np.mean(head_recall))
    print(f'{"head truncation":24}{base:>16.3f}{"--":>9}{0:>10.0f}', flush=True)

    for label, embedder in [('TF-IDF (lexical)', None),
                            ('all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2'),
                            ('TechWolf/JobBERT-v2', 'TechWolf/JobBERT-v2')]:
        try:
            select_relevant(rows[0]['resume_text'], rows[0]['job_description_text'], budget, embedder)
        except Exception as exc:
            print(f'{label:24}{"FAILED":>16}   {type(exc).__name__}: {exc}', flush=True)
            continue

        recall, start = [], time.perf_counter()
        for r in rows:
            jd_terms = content_words(r['job_description_text'])
            sel = content_words(select_relevant(r['resume_text'], r['job_description_text'], budget, embedder))
            recall.append(len(jd_terms & sel) / max(len(jd_terms), 1))
        ms = (time.perf_counter() - start) / len(rows) * 1000
        mean = float(np.mean(recall))
        print(f'{label:24}{mean:>16.3f}{mean-base:>+9.3f}{ms:>10.0f}', flush=True)


if __name__ == '__main__':
    main()
