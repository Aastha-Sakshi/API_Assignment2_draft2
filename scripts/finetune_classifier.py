"""
Assignment requirement 8 — fine-tune a model on a domain dataset.

Dataset : cnamuangtoun/resume-job-description-fit  (HuggingFace Hub)
          6,241 train rows + a held-out test split
          columns: resume_text | job_description_text | label
          labels : No Fit (3143) · Potential Fit (1556) · Good Fit (1542)
Base    : distilbert-base-uncased (66M params — trains on a Colab T4 in
          minutes and runs on CPU at inference, which is the deploy target)

The dataset is imbalanced ~50/25/25, so:
  * the loss is class-weighted, otherwise the model collapses onto "No Fit"
    and reports a flattering 50% accuracy while having learned nothing;
  * the headline metric is macro-F1, not accuracy.

Run locally:
    python scripts/finetune_classifier.py --epochs 3

Run on Colab (GPU):
    !pip install -q transformers datasets accelerate scikit-learn
    !python scripts/finetune_classifier.py --epochs 3 --batch-size 32 --fp16
    # then zip models/finetuned-fit-classifier and download it
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "finetuned-fit-classifier"
LABELS = ["No Fit", "Potential Fit", "Good Fit"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for resume-JD fit classification")
    parser.add_argument("--dataset", default="cnamuangtoun/resume-job-description-fit")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 = use all rows")
    # On CPU an eval pass over the full test split costs ~13 min, and it runs
    # once per epoch plus twice more. Subsampling it is the difference between
    # a 2-hour local run and a 4-hour one. Always 0 (full split) on GPU.
    parser.add_argument("--eval-rows", type=int, default=0, help="0 = full test split")
    # The ablation this script exists to settle: does spending the same token
    # budget on JD-relevant resume content beat spending it on the first N
    # tokens? Must match app/config.py at serve time or the model is served a
    # different input distribution than it was trained on.
    parser.add_argument("--selection", default="head", choices=["head", "jd_guided"],
                        help="head = plain truncation; jd_guided = app/selection.py")
    parser.add_argument("--fp16", action="store_true", help="GPU only")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision,
        "macro_recall": recall,
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def _load_selection():
    """Import selection whether running from the repo (app.selection) or from a
    flat Colab working directory (selection.py written next to this script)."""
    import sys
    from pathlib import Path

    # Try every layout this script legitimately runs under: the repo (app.selection),
    # beside itself (Colab writes scripts/selection.py), and the working directory.
    for candidate in (Path(__file__).resolve().parent, Path.cwd()):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        from app import selection
    except ImportError:
        import selection
    return selection


def main():
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    args = parse_args()
    set_seed(args.seed)

    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Loading dataset {args.dataset} ...")
    raw = load_dataset(args.dataset)
    train_split, eval_split = raw["train"], raw["test"]

    if args.max_train_rows:
        train_split = train_split.shuffle(seed=args.seed).select(
            range(min(args.max_train_rows, len(train_split)))
        )
    if args.eval_rows:
        eval_split = eval_split.shuffle(seed=args.seed).select(
            range(min(args.eval_rows, len(eval_split)))
        )

    print(f"train={len(train_split)}  eval={len(eval_split)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    selection = _load_selection() if args.selection == "jd_guided" else None

    def preprocess(batch):
        jds, resumes = batch["job_description_text"], batch["resume_text"]

        if selection is not None:
            pairs = [
                selection.prepare_pair(resume, jd, max_tokens=args.max_length, embedder=None)
                for jd, resume in zip(jds, resumes)
            ]
            jds = [p[0] for p in pairs]
            resumes = [p[1] for p in pairs]

        # JD first: it is the shorter field, so truncation trims the resume
        # tail rather than dropping the requirements we match against.
        encoded = tokenizer(jds, resumes, truncation=True, max_length=args.max_length)
        encoded["labels"] = [label2id[label] for label in batch["label"]]
        return encoded

    drop = train_split.column_names
    train_ds = train_split.map(preprocess, batched=True, remove_columns=drop)
    eval_ds = eval_split.map(preprocess, batched=True, remove_columns=drop)

    counts = np.bincount(np.array(train_ds["labels"]), minlength=len(LABELS))
    class_weights = torch.tensor(
        counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float
    )
    print("label counts:", dict(zip(LABELS, counts.tolist())))
    print("class weights:", [round(w, 3) for w in class_weights.tolist()])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir).parent / "training-checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=args.fp16,
        report_to=[],
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Evaluating before training gives the report an honest "before" column:
    # an untrained classification head, i.e. chance-level performance.
    print("\n=== Before fine-tuning (untrained head) ===")
    baseline = trainer.evaluate()
    print(json.dumps({k: round(v, 4) for k, v in baseline.items() if isinstance(v, float)}, indent=2))

    trainer.train()

    print("\n=== After fine-tuning ===")
    final = trainer.evaluate()
    print(json.dumps({k: round(v, 4) for k, v in final.items() if isinstance(v, float)}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "base_model": args.base_model,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "selection": args.selection,
                "learning_rate": args.lr,
                "train_rows": len(train_ds),
                "eval_rows": len(eval_ds),
                "label_counts": dict(zip(LABELS, counts.tolist())),
                "before_finetuning": {k: v for k, v in baseline.items() if isinstance(v, float)},
                "after_finetuning": {k: v for k, v in final.items() if isinstance(v, float)},
                "log_history": trainer.state.log_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to {output_dir}")
    print("training_report.json holds the before/after numbers and the per-epoch curve.")


if __name__ == "__main__":
    main()
