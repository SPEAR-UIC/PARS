import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess lmsys/lmsys-chat-1m (model=gpt-4, turn=1) and generate pairwise train/val data."
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and pairing.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--similarity-threshold", type=float, default=0.2, help="Minimum relative length difference.")
    parser.add_argument("--train-num-pairs", type=int, default=50, help="Pairs per sample for train split.")
    parser.add_argument("--val-num-pairs", type=int, default=5, help="Pairs per sample for val split.")
    return parser.parse_args()


def filter_gpt4_turn1(example):
    return example.get("model") == "gpt-4" and example.get("turn") == 1


def normalize(split_data):
    rows = []
    for idx, item in enumerate(split_data):
        conversation = item.get("conversation", [])
        if not conversation:
            continue
        prompt = conversation[0].get("content", "")
        answer = conversation[-1].get("content", "")
        if not prompt or not answer:
            continue
        rows.append(
            {
                "task_id": idx,
                "prompt": prompt,
                "answer_content": answer,
                "answer_length": len(answer),
            }
        )
    return rows


def make_pairs(rows, num_pairs_per_sample, similarity_threshold, seed):
    rng = random.Random(seed)
    pairs = []
    n = len(rows)
    max_attempts = 10 * num_pairs_per_sample

    for i, sample_a in tqdm(enumerate(rows), total=n, desc="Generating pairs"):
        generated = 0
        attempts = 0
        while generated < num_pairs_per_sample and attempts < max_attempts:
            attempts += 1
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            sample_b = rows[j]
            len_a = sample_a["answer_length"]
            len_b = sample_b["answer_length"]
            diff = abs(len_a - len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0
            if diff < similarity_threshold:
                continue
            pairs.append(
                {
                    "prompt_A": sample_a["prompt"],
                    "prompt_B": sample_b["prompt"],
                    "label": 1 if len_a > len_b else 0,
                }
            )
            generated += 1
    return pairs


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmsys/lmsys-chat-1m")["train"]
    ds = ds.filter(filter_gpt4_turn1)
    ds_split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train_rows = normalize(ds_split["train"])
    val_rows = normalize(ds_split["test"])

    train_pairs = make_pairs(
        rows=train_rows,
        num_pairs_per_sample=args.train_num_pairs,
        similarity_threshold=args.similarity_threshold,
        seed=args.seed,
    )
    val_pairs = make_pairs(
        rows=val_rows,
        num_pairs_per_sample=args.val_num_pairs,
        similarity_threshold=args.similarity_threshold,
        seed=args.seed + 1,
    )

    save_json(train_rows, out_dir / "train_data.json")
    save_json(val_rows, out_dir / "val_data.json")
    save_json(train_pairs, out_dir / f"train_pairs_length_diff_{args.similarity_threshold}.json")
    save_json(val_pairs, out_dir / f"val_pairs_length_diff_{args.similarity_threshold}.json")

    print(f"Saved: {out_dir / 'train_data.json'} ({len(train_rows)})")
    print(f"Saved: {out_dir / 'val_data.json'} ({len(val_rows)})")
    print(f"Saved: {out_dir / f'train_pairs_length_diff_{args.similarity_threshold}.json'} ({len(train_pairs)})")
    print(f"Saved: {out_dir / f'val_pairs_length_diff_{args.similarity_threshold}.json'} ({len(val_pairs)})")


if __name__ == "__main__":
    main()
