"""Fine-tune distilbert-base-german-cased on 10kGNAD.

    python scripts/train_transformer.py              # full run (about 15 min on a Colab T4 GPU)
    python scripts/train_transformer.py --smoke      # 20 steps on 200 articles, to test the pipeline

Picks the epoch with the best validation macro-F1, then scores it once on the test split.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnad.data import SEED, load  # noqa: E402
from gnad.metrics import evaluate  # noqa: E402

BASE_MODEL = "distilbert-base-german-cased"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    set_seed(SEED)

    data = load()
    names = data.label_names
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_dataset(split, limit=None):
        texts, labels = split.texts[:limit], split.labels[:limit]
        ds = Dataset.from_dict({"text": texts, "label": labels})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length), batched=True, remove_columns=["text"])

    limit = 200 if args.smoke else None
    train, val, test = to_dataset(data.train, limit), to_dataset(data.validation, limit), to_dataset(data.test, limit)
    truncated = sum(len(ids) >= args.max_length for ids in to_dataset(data.test)["input_ids"]) if not args.smoke else None

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(names), id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)}
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        result = evaluate(labels.tolist(), np.argmax(logits, axis=-1).tolist(), names)
        return {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}

    out_dir = ROOT / ("checkpoints/smoke" if args.smoke else "checkpoints/distilbert")
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=20 if args.smoke else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch 16, fits in 8 GB unified memory
        per_device_eval_batch_size=16,
        warmup_steps=0.1,  # a float below 1 is a ratio of the total steps
        weight_decay=0.01,
        eval_strategy="steps" if args.smoke else "epoch",
        eval_steps=10 if args.smoke else None,
        save_strategy="steps" if args.smoke else "epoch",
        save_steps=10 if args.smoke else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        train_sampling_strategy="group_by_length",  # batches of similar length waste less padding
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),  # half precision on an NVIDIA GPU (Colab); MPS and CPU stay fp32
        seed=SEED,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60

    start = time.perf_counter()
    predictions = trainer.predict(test)
    test_seconds = time.perf_counter() - start
    pred = np.argmax(predictions.predictions, axis=-1).tolist()
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

    result = evaluate(test["label"], pred, names)
    result.update(
        model=BASE_MODEL, device=str(trainer.args.device), max_length=args.max_length, epochs=args.epochs,
        learning_rate=args.lr, train_minutes=round(train_minutes, 1), test_seconds=round(test_seconds, 1),
        truncated_test_articles=truncated,
        validation_history=[h for h in trainer.state.log_history if "eval_macro_f1" in h],
        best_checkpoint=trainer.state.best_model_checkpoint,
    )
    tag = "transformer_smoke" if args.smoke else "transformer"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(ROOT / "results" / f"{tag}.test_predictions.jsonl", "w") as f:
        for i, (gold, p, pr) in enumerate(zip(test["label"], pred, probs)):
            f.write(json.dumps({"index": i, "label": gold, "pred": p, "confidence": round(float(pr.max()), 4)}) + "\n")

    if not args.smoke:
        save_dir = ROOT / "models" / "distilbert"
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
    print(f"{tag}: accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}  ({train_minutes:.1f} min on {trainer.args.device})")


if __name__ == "__main__":
    main()
