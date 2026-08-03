"""
Generate finetune_colab.ipynb, embedding the training script and the
selection module.

    python scripts/build_colab_notebook.py

Why generated rather than hand-written: `google.colab.files.upload()` only
works in the Colab browser frontend, so a notebook driven from the VS Code
Colab extension has no way to receive a local file. The sources therefore have
to be inlined via %%writefile — and inlined copies rot the moment the real
files change. Generating means re-running this script is the only sync step,
and a stale notebook is a one-command fix rather than a debugging session on
a GPU runtime.

Re-run after editing scripts/finetune_classifier.py or app/selection.py.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "finetune_classifier.py"
SELECTION = ROOT / "app" / "selection.py"
TARGET = ROOT / "finetune_colab.ipynb"


def code(body: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in body.strip("\n").split("\n")],
    }


def markdown(body: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in body.strip("\n").split("\n")],
    }


def embed(magic: str, path: Path):
    text = path.read_text(encoding="utf-8")
    if "%%writefile" in text:
        raise ValueError(f"{path.name} contains a cell magic; would break the notebook")
    return code(f"%%writefile {magic}\n" + text)


# The ablation table. Written with .format() and single-quoted keys so it
# parses on any Python 3.8+; nested quotes inside f-strings need 3.12.
ABLATION_CELL = """
# The ablation table -- this is the report screenshot.
import json
import shutil
from pathlib import Path

arms = {'head truncation': 'models/fit-head',
        'JD-guided selection': 'models/fit-jd-guided'}

reports = {}
for name, directory in arms.items():
    path = Path(directory) / 'training_report.json'
    if path.exists():
        reports[name] = json.loads(path.read_text())
    else:
        print('MISSING {} -- did that training cell fail?'.format(path))

if not reports:
    raise SystemExit('No training reports found. Re-run the two training cells.')

row = '{:<22}{:>10.4f}{:>10.4f}{:>13.4f}'
print('{:<22}{:>10}{:>10}{:>13}'.format('arm', 'accuracy', 'macro_f1', 'weighted_f1'))

base = next(iter(reports.values()))['before_finetuning']
print(row.format('untrained head', base['eval_accuracy'],
                 base['eval_macro_f1'], base['eval_weighted_f1']))

for name, report in reports.items():
    after = report['after_finetuning']
    print(row.format(name, after['eval_accuracy'],
                     after['eval_macro_f1'], after['eval_weighted_f1']))

# Ship whichever arm won as the model the application serves. Deciding here,
# on the numbers, avoids a human copying the wrong directory later.
winner = max(reports, key=lambda k: reports[k]['after_finetuning']['eval_macro_f1'])
print('\\nwinner: ' + winner)

shutil.rmtree('models/finetuned-fit-classifier', ignore_errors=True)
shutil.copytree(arms[winner], 'models/finetuned-fit-classifier')
Path('models/finetuned-fit-classifier/selection_strategy.txt').write_text(
    reports[winner]['selection'])
print('copied to models/finetuned-fit-classifier')
"""


def build():
    notebook = {
        "cells": [
            markdown("""
# Fine-tuning the resume-fit classifier

**Assignment requirement 8** — fine-tune a model on a domain dataset.

This notebook trains DistilBERT to classify a (resume, job description) pair as
*No Fit* / *Potential Fit* / *Good Fit*, on 6,241 rows of
`cnamuangtoun/resume-job-description-fit`.

It trains **two arms** that differ in one variable — how the 512 input tokens
are chosen — so the choice can be made on evidence rather than intuition:

| Arm | Input preparation |
|---|---|
| A | **Head truncation** — keep the first 512 tokens |
| B | **JD-guided selection** — keep the resume blocks most relevant to the job description |

The last cells score both, copy the winner into
`models/finetuned-fit-classifier/`, and package everything as `finetuned.zip`
for download.

---

### Before running

**Runtime → Change runtime type → T4 GPU**, then *Run all*.
About 20 minutes for both arms on a T4, against ~8 hours on a laptop CPU.

Works in the Colab browser and through the VS Code Colab extension alike: no
`files.upload()` / `files.download()` widgets are used, as those only function
in a real browser session.

> Cells marked *generated* are written from `scripts/finetune_classifier.py`
> and `app/selection.py`. Do not edit them here — edit those files and re-run
> `python scripts/build_colab_notebook.py`.
"""),
            markdown("## Step 0 — environment"),
            code("""
# Confirm a GPU is attached. If this errors, fix the runtime type first.
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
            code("""
# transformers is pinned to the version the application serves with, so the
# saved model loads locally without a version surprise.
!pip install -q "transformers==4.44.2" datasets accelerate scikit-learn
"""),
            markdown("""
## Step 1 — write the training code onto the VM *(generated)*

Both files go in `scripts/`, and the directory matters twice over. The training
script derives its default output path from `Path(__file__).parent.parent`,
which from `/content` would resolve to `/` and save the model outside the
working directory. And `python scripts/finetune_classifier.py` puts `scripts/`
on `sys.path` — not the working directory — so a copy of `selection.py` at
`/content` is invisible to the import.

`selection.py` is the same module the running application uses. If the two ever
diverge, the served model gets an input distribution it never trained on and
degrades silently.
"""),
            code("""
import os

os.makedirs('scripts', exist_ok=True)
"""),
            embed("scripts/finetune_classifier.py", TRAINER),
            embed("scripts/selection.py", SELECTION),
            markdown("""
## Step 2 — train both arms

Identical in every respect except input preparation, so any difference between
them is attributable to that one variable.
"""),
            code("""
# ARM A -- head truncation (the baseline).
#
#     --max-length 512 is DistilBERT's ceiling and it matters here: the median
#     resume+JD pair is ~1,460 tokens, so 384 fed the model 26% of each pair
#     and 512 raises that to 34%. Truncation, not capacity, is this model's
#     binding constraint.
#
#     --epochs 6 because macro-F1 was still climbing at epoch 3
#     (0.248 -> 0.347 -> 0.380): the first run stopped short of converging.
!python scripts/finetune_classifier.py \\
    --epochs 6 --batch-size 16 --lr 3e-5 --max-length 512 --fp16 \\
    --selection head --output-dir models/fit-head
"""),
            code("""
# ARM B -- JD-guided selection.
#     Preprocessing costs ~11 ms/row (TF-IDF), about 90s over the dataset.
!python scripts/finetune_classifier.py \\
    --epochs 6 --batch-size 16 --lr 3e-5 --max-length 512 --fp16 \\
    --selection jd_guided --output-dir models/fit-jd-guided
"""),
            markdown("""
## Step 3 — score both arms and ship the winner

The comparison decides which checkpoint the application serves, and it is made
here on macro-F1 rather than by a human copying a directory later. The chosen
strategy is written to `selection_strategy.txt` inside the model directory, and
the serving code reads it instead of assuming — a model trained on one input
preparation and served another degrades without erroring.
"""),
            code(ABLATION_CELL),
            markdown("## Step 4 — package for download"),
            code("""
# Package the winning model plus BOTH training reports, so the ablation can
#    be written up locally without needing the runtime again. The reports are
#    globbed rather than named: a failed arm then costs one missing row in the
#    table instead of aborting the zip and stranding the model on the runtime.
!zip -rq finetuned.zip models/finetuned-fit-classifier
!zip -rq finetuned.zip models/fit-*/training_report.json
!ls -lh finetuned.zip
!unzip -l finetuned.zip | grep training_report
"""),
            markdown("""
## Getting `finetuned.zip` back to the project

**VS Code Colab extension:** open the Colab icon in the Activity Bar, find
`finetuned.zip` in the *Contents* view, right-click -> Download.

**Colab in a browser:** run `from google.colab import files;
files.download('finetuned.zip')` in a new cell.

Then locally:

```bash
unzip finetuned.zip -d .          # creates models/finetuned-fit-classifier/
python scripts/smoke_test.py      # classify_fit_finetuned should PASS
python scripts/evaluate.py --limit 100
```

`evaluate.py` writes `logs/eval_report.json` — the three-arm head-to-head,
also served as metric **M7** on `GET /metrics`.
"""),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.name}: {len(notebook['cells'])} cells")
    print(f"  embedded {TRAINER.relative_to(ROOT)} ({len(TRAINER.read_text(encoding='utf-8').splitlines())} lines)")
    print(f"  embedded {SELECTION.relative_to(ROOT)} ({len(SELECTION.read_text(encoding='utf-8').splitlines())} lines)")


if __name__ == "__main__":
    build()
