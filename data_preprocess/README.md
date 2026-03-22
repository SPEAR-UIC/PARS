# Data Preprocess

This directory contains one preprocessing script per dataset.
Each script downloads the dataset, applies dataset-specific filtering/selection
based on the source notebook logic, splits train/val, and generates pairwise
classification samples.

## Model note

The four dataset pipelines in this folder are GPT4-based data preparation paths.
If you need LLaMA, R1, or other model-generated variants, please generate those
datasets manually.

## Scripts

- `scripts/preprocess_alpaca_gpt4.py` for `vicgalle/alpaca-gpt4`
- `scripts/preprocess_code_gpt4.py` for `theblackcat102/evol-codealpaca-v1`
- `scripts/preprocess_lmsys_gpt4.py` for `lmsys/lmsys-chat-1m` with `model=gpt-4` and `turn=1`
- `scripts/preprocess_math_gpt4.py` for `TIGER-Lab/MATH-plus`

## Install

```bash
pip install datasets tqdm
```

## Run

```bash
python scripts/preprocess_alpaca_gpt4.py --output-dir outputs/alpaca/gpt4
python scripts/preprocess_code_gpt4.py --output-dir outputs/code/gpt4
python scripts/preprocess_lmsys_gpt4.py --output-dir outputs/lmsys/gpt4
python scripts/preprocess_math_gpt4.py --output-dir outputs/math/gpt4
```

Each script writes:

- `train_data.json`
- `val_data.json`
- `train_pairs_length_diff_<threshold>.json`
- `val_pairs_length_diff_<threshold>.json`

The default pairwise `threshold` is `0.2` for all four scripts.
