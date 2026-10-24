"""Fine-tune a transformer to classify movie genre from plot text.

Dataset: Wikipedia Movie Plots (https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots),
a CSV with `Genre` and `Plot` columns. Save it as data/wiki_movie_plots_deduped.csv.

Run from the ScriptAgentService directory:
    python -m training.train_genre_classifier --csv data/wiki_movie_plots_deduped.csv

A GPU (e.g. a free Colab T4) trains the default config in ~15 minutes; CPU works
for a quick smoke test with --max-per-class 100 --epochs 1.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from app.genres import parse_genres


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("models/genre-classifier"))
    p.add_argument("--base-model", default="distilbert-base-uncased")
    p.add_argument("--max-per-class", type=int, default=1500,
                   help="Cap examples per genre to reduce drama/comedy imbalance.")
    p.add_argument("--min-per-class", type=int, default=200,
                   help="Drop genres with fewer examples than this.")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(csv: Path, max_per_class: int, min_per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv, usecols=["Genre", "Plot"]).dropna()
    df["label"] = df["Genre"].map(lambda g: next(iter(parse_genres(g)), None))
    df = df.dropna(subset=["label"])

    counts = df["label"].value_counts()
    df = df[df["label"].isin(counts[counts >= min_per_class].index)]
    df = (
        df.sample(frac=1, random_state=seed)
        .groupby("label")
        .head(max_per_class)
        .reset_index(drop=True)
    )
    print("Examples per genre:\n" + df["label"].value_counts().to_string())
    return df


class PlotDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


@torch.inference_mode()
def evaluate(model, loader, device, use_amp: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, preds, golds = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(**batch)
        total_loss += out.loss.item() * batch["labels"].size(0)
        preds.append(out.logits.argmax(-1).cpu().numpy())
        golds.append(batch["labels"].cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(golds)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    df = load_dataset(args.csv, args.max_per_class, args.min_per_class, args.seed)
    labels = sorted(df["label"].unique())
    label2id = {g: i for i, g in enumerate(labels)}
    id2label = {i: g for g, i in label2id.items()}
    y = df["label"].map(label2id).tolist()

    x_train, x_val, y_train, y_val = train_test_split(
        df["Plot"].tolist(), y, test_size=0.15, stratify=y, random_state=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(PlotDataset(x_train, y_train), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PlotDataset(x_val, y_val), batch_size=args.batch_size * 2,
                            collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight")
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running_loss = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running_loss / step:.4f}")

        val_loss, preds, golds = evaluate(model, val_loader, device, use_amp)
        macro_f1 = f1_score(golds, preds, average="macro")
        accuracy = float((preds == golds).mean())
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader),
                        "val_loss": val_loss, "val_accuracy": accuracy, "val_macro_f1": macro_f1})
        print(f"epoch {epoch} done in {time.time() - start:.0f}s | val_loss {val_loss:.4f} "
              f"| acc {accuracy:.3f} | macro-F1 {macro_f1:.3f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            report = classification_report(golds, preds, target_names=labels, output_dict=True)
            print(classification_report(golds, preds, target_names=labels, digits=3))

    metrics = {"base_model": args.base_model, "labels": labels, "best_val_macro_f1": best_f1,
               "history": history, "per_class_report": report}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved best model (macro-F1 {best_f1:.3f}) to {args.out}")


if __name__ == "__main__":
    main()
