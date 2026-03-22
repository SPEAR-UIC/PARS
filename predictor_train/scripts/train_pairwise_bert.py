import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT-based pairwise ranker from pairwise JSON data."
    )
    parser.add_argument("--train-file", required=True, help="Path to train pairwise JSON.")
    parser.add_argument("--val-file", required=True, help="Path to validation pairwise JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--model-name", default="bert-base-uncased", help="Hugging Face encoder name.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=1.0, help="MarginRankingLoss margin.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairwisePromptDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt_a = sample["prompt_A"]
        prompt_b = sample["prompt_B"]
        label = float(sample["label"])

        encoding_a = self.tokenizer(
            prompt_a,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding_b = self.tokenizer(
            prompt_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids_A": encoding_a["input_ids"].squeeze(0),
            "attention_mask_A": encoding_a["attention_mask"].squeeze(0),
            "input_ids_B": encoding_b["input_ids"].squeeze(0),
            "attention_mask_B": encoding_b["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "input_ids_A": torch.stack([item["input_ids_A"] for item in batch]),
        "attention_mask_A": torch.stack([item["attention_mask_A"] for item in batch]),
        "input_ids_B": torch.stack([item["input_ids_B"] for item in batch]),
        "attention_mask_B": torch.stack([item["attention_mask_B"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


class PairwiseRanker(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def score(self, input_ids, attention_mask):
        hidden = self.encode(input_ids, attention_mask)
        return self.fc(hidden).squeeze(-1)

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        score_a = self.score(input_ids_A, attention_mask_A)
        score_b = self.score(input_ids_B, attention_mask_B)
        return score_a, score_b


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(model, dataloader, device, criterion, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids_a = batch["input_ids_A"].to(device)
            attention_mask_a = batch["attention_mask_A"].to(device)
            input_ids_b = batch["input_ids_B"].to(device)
            attention_mask_b = batch["attention_mask_B"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                score_a, score_b = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
                target = 2 * labels - 1
                loss = criterion(score_a, score_b, target)

            total_loss += loss.item()
            predictions = (score_a > score_b).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def save_checkpoint(model, output_path):
    torch.save(model.state_dict(), output_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_json(args.train_file)
    val_samples = load_json(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = PairwisePromptDataset(train_samples, tokenizer, args.max_length)
    val_dataset = PairwisePromptDataset(val_samples, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = PairwiseRanker(args.model_name).to(device)
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MarginRankingLoss(margin=args.margin)
    num_training_steps = args.num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_accuracy = -1.0
    history = []

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress:
            input_ids_a = batch["input_ids_A"].to(device)
            attention_mask_a = batch["attention_mask_A"].to(device)
            input_ids_b = batch["input_ids_B"].to(device)
            attention_mask_b = batch["attention_mask_B"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                score_a, score_b = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
                target = 2 * labels - 1
                loss = criterion(score_a, score_b, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion, use_amp)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, output_dir / "best_model.pt")

    save_checkpoint(model, output_dir / "last_model.pt")

    metrics = {
        "train_file": args.train_file,
        "val_file": args.val_file,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "max_length": args.max_length,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved best checkpoint to {output_dir / 'best_model.pt'}")
    print(f"Saved last checkpoint to {output_dir / 'last_model.pt'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
