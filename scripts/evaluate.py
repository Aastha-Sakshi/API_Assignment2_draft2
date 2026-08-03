"""
Head-to-head evaluation on the held-out test split.

    Arm A  fine-tuned DistilBERT       512-token budget   (requirement 8)
    Arm B  prompted gpt-oss-20b          full document      (zero-shot control)
    Arm C  prompted gpt-oss-20b          same 512 tokens as Arm A

Arm C exists because A-vs-B confounds two variables at once: the LLM has ~300x
more parameters AND reads ~3x more of the document. With C the effects
separate:

    B - C   how much the extra context is worth
    C - A   how much the extra capability is worth

Without C, "the LLM won" is an unattributable result. Output feeds metric M7 on GET /metrics.

    python scripts/evaluate.py --limit 100
    python scripts/evaluate.py --limit 100 --arms finetuned   # skip API calls

Writes logs/eval_report.json.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, pipeline  # noqa: E402

LABELS = config.FIT_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned vs prompted fit classification")
    parser.add_argument("--dataset", default=config.FIT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100, help="rows to evaluate (stratified)")
    parser.add_argument("--arms", default="all",
                        choices=["all", "finetuned", "prompted", "prompted_matched", "local"],
                        help="local = finetuned only (no API calls)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(config.LOG_DIR / "eval_report.json"))
    return parser.parse_args()


def stratified_sample(rows, limit, seed):
    """Even split across labels, so a majority-class-heavy sample cannot
    flatter either arm."""
    import random

    rng = random.Random(seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    per_label = max(1, limit // len(by_label))
    sample = []
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        sample.extend(pool[:per_label])
    rng.shuffle(sample)
    return sample[:limit]


def score(y_true, y_pred, latencies, unparsed=0):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0), 4),
        "per_class_f1": {label: round(report[label]["f1-score"], 4) for label in LABELS},
        "confusion_matrix": {
            "labels": LABELS,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
        "prediction_distribution": dict(Counter(y_pred)),
        "mean_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "unparsable_outputs": unparsed,
    }


def main():
    from datasets import load_dataset

    args = parse_args()
    print(f"Loading {args.dataset}[{args.split}] ...")
    rows = list(load_dataset(args.dataset, split=args.split))
    sample = stratified_sample(rows, args.limit, args.seed)
    print(f"Evaluating {len(sample)} rows: {dict(Counter(r['label'] for r in sample))}\n")

    y_true = [r["label"] for r in sample]
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "n_evaluated": len(sample),
        "true_distribution": dict(Counter(y_true)),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": {},
    }

    run_ft = args.arms in ("all", "finetuned", "local")
    run_pr = args.arms in ("all", "prompted")
    run_pm = args.arms in ("all", "prompted_matched")

    if run_ft:
        if not pipeline.finetuned_available():
            print(f"SKIP fine-tuned arm: no model at {config.FINETUNED_DIR}")
            print("     Run scripts/finetune_classifier.py first.")
        else:
            preds, latencies = [], []
            for i, row in enumerate(sample, 1):
                result = pipeline.classify_fit_finetuned(row["resume_text"], row["job_description_text"])
                preds.append(result["label"])
                latencies.append(result["latency_sec"])
                if i % 20 == 0:
                    print(f"  fine-tuned {i}/{len(sample)}")
            report["arms"]["finetuned"] = {
                "model": config.MODEL_REGISTRY["fit_finetuned"]["model"],
                **score(y_true, preds, latencies),
            }

    def run_llm_arm(arm_name, matched):
        preds, latencies, tokens, unparsed, chars, degraded = [], [], 0, 0, 0, 0
        for i, row in enumerate(sample, 1):
            result = pipeline.classify_fit_prompted(
                row["resume_text"], row["job_description_text"], match_slm_input=matched
            )
            # A degraded row was answered by flan-t5-base, not gpt-oss-20b.
            # Scoring it would credit or blame the wrong model, so it counts as
            # unparsed: still a wrong answer (never silently dropped), but never
            # attributed to the model this arm is supposed to measure.
            if result["degraded"]:
                degraded += 1
                unparsed += 1
                preds.append("UNPARSED")
            # An unparsable answer is a wrong answer, not a dropped row --
            # discarding it would inflate the prompted arm's score.
            elif result["parse_failed"]:
                unparsed += 1
                preds.append("UNPARSED")
            else:
                preds.append(result["label"])
            latencies.append(result["latency_sec"])
            tokens += result.get("total_tokens") or 0
            chars += result.get("input_chars") or 0
            if i % 20 == 0:
                print(f"  {arm_name} {i}/{len(sample)}")
        report["arms"][arm_name] = {
            "model": config.NVIDIA_MODEL,
            "input": "512-token matched" if matched else "full document",
            "mean_input_chars": round(chars / len(sample)),
            "total_tokens": tokens,
            "estimated_cost_usd": round(tokens / 1_000_000 * 0.10, 6),
            "degraded_rows": degraded,
            **score(y_true, preds, latencies, unparsed),
        }
        if degraded:
            print(f"  WARNING: {degraded}/{len(sample)} rows in {arm_name} fell back to "
                  f"{config.FALLBACK_GEN_MODEL} and were scored as unparsed.")

    if run_pr or run_pm:
        if not config.llm_available():
            print("SKIP prompted arms: NVIDIA_API_KEY not set (a local-fallback")
            print("     comparison would measure flan-t5-base, not gpt-oss-20b).")
        else:
            if run_pr:
                run_llm_arm("prompted_full", matched=False)
            if run_pm:
                run_llm_arm("prompted_matched", matched=True)

    arms = report["arms"]
    verdict = {}

    if "finetuned" in arms and "prompted_full" in arms:
        ft, full = arms["finetuned"], arms["prompted_full"]
        verdict["winner"] = "finetuned" if ft["macro_f1"] >= full["macro_f1"] else "prompted_full"
        verdict["macro_f1_finetuned_minus_prompted_full"] = round(ft["macro_f1"] - full["macro_f1"], 4)
        if ft["mean_latency_sec"]:
            verdict["latency_speedup_x"] = round(full["mean_latency_sec"] / ft["mean_latency_sec"], 1)

    # The decomposition Arm C exists for.
    if "prompted_full" in arms and "prompted_matched" in arms:
        verdict["context_effect_macro_f1"] = round(
            arms["prompted_full"]["macro_f1"] - arms["prompted_matched"]["macro_f1"], 4
        )
    if "prompted_matched" in arms and "finetuned" in arms:
        verdict["capability_effect_macro_f1"] = round(
            arms["prompted_matched"]["macro_f1"] - arms["finetuned"]["macro_f1"], 4
        )
    if "context_effect_macro_f1" in verdict and "capability_effect_macro_f1" in verdict:
        verdict["reading"] = (
            "context_effect = what the extra document access buys the LLM; "
            "capability_effect = what the extra parameters buy it at equal input. "
            "A large context_effect means the fine-tuned model is limited by its "
            "512-token window rather than by its size."
        )

    if verdict:
        report["verdict"] = verdict

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + json.dumps(report.get("verdict", report["arms"]), indent=2))
    print(f"\nWritten to {out_path} — now visible as M7 on GET /metrics")


if __name__ == "__main__":
    main()
