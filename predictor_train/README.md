# Predictor Train

This directory contains the training code for the pairwise prompt ranker.

The released training path is a BERT-based pairwise ranking model trained with
margin ranking loss on JSON files of the form:

```json
{
  "prompt_A": "...",
  "prompt_B": "...",
  "label": 1
}
```

`label=1` means `prompt_A` is expected to produce a longer response than
`prompt_B`. `label=0` means the opposite.

The training script consumes the pairwise outputs produced by
`../data_preprocess/`.

## Install

```bash
pip install -r requirements.txt
```

## Input files

Expected input files:

- `train_file`: pairwise training JSON
- `val_file`: pairwise validation JSON

Default data-preprocess output locations:

- Alpaca GPT4 train: `../data_preprocess/outputs/alpaca/gpt4/train_pairs_length_diff_0.2.json`
- Alpaca GPT4 val: `../data_preprocess/outputs/alpaca/gpt4/val_pairs_length_diff_0.2.json`
- Code GPT4 train: `../data_preprocess/outputs/code/gpt4/train_pairs_length_diff_0.2.json`
- Code GPT4 val: `../data_preprocess/outputs/code/gpt4/val_pairs_length_diff_0.2.json`
- LMSYS GPT4 train: `../data_preprocess/outputs/lmsys/gpt4/train_pairs_length_diff_0.2.json`
- LMSYS GPT4 val: `../data_preprocess/outputs/lmsys/gpt4/val_pairs_length_diff_0.2.json`
- Math GPT4 train: `../data_preprocess/outputs/math/gpt4/train_pairs_length_diff_0.2.json`
- Math GPT4 val: `../data_preprocess/outputs/math/gpt4/val_pairs_length_diff_0.2.json`

## Run examples

Train on Alpaca GPT4:

```bash
python scripts/train_pairwise_bert.py \
  --train-file ../data_preprocess/outputs/alpaca/gpt4/train_pairs_length_diff_0.2.json \
  --val-file ../data_preprocess/outputs/alpaca/gpt4/val_pairs_length_diff_0.2.json \
  --output-dir outputs/alpaca_gpt4_bert
```

Train on LMSYS GPT4:

```bash
python scripts/train_pairwise_bert.py \
  --train-file ../data_preprocess/outputs/lmsys/gpt4/train_pairs_length_diff_0.2.json \
  --val-file ../data_preprocess/outputs/lmsys/gpt4/val_pairs_length_diff_0.2.json \
  --output-dir outputs/lmsys_gpt4_bert
```

## Options

- `--model-name bert-base-uncased`
- `--batch-size 128`
- `--num-epochs 3`
- `--learning-rate 2e-5`
- `--max-length 128`
- `--margin 1.0`
- `--warmup-ratio 0.1`
- `--weight-decay 0.01`
- `--num-workers 0`
- `--seed 42`
- `--resume-from path/to/checkpoint.pt`

Option details:

- `--train-file`: path to training pairwise JSON.
- `--val-file`: path to validation pairwise JSON.
- `--output-dir`: directory used to save checkpoints and metrics.
- `--model-name`: Hugging Face encoder name. Default is `bert-base-uncased`.
- `--max-length`: tokenizer truncation and padding length.
- `--batch-size`: batch size for both train and validation.
- `--num-epochs`: total training epochs. Default is `3`.
- `--learning-rate`: AdamW learning rate.
- `--margin`: margin used by `MarginRankingLoss`.
- `--warmup-ratio`: fraction of training steps used for scheduler warmup.
- `--weight-decay`: AdamW weight decay.
- `--num-workers`: PyTorch dataloader worker count.
- `--seed`: random seed for reproducibility.
- `--resume-from`: checkpoint path for continued training.

Outputs:

- `best_model.pt`
- `last_model.pt`
- `metrics.json`

Output details:

- `best_model.pt`: the checkpoint with the best validation accuracy during training. This is usually the safest file to use for later evaluation or serving.
- `last_model.pt`: the checkpoint saved at the end of the final epoch. This is the final training state, even if its validation accuracy is not the best one.
- `metrics.json`: training metadata and per-epoch results, including `train_loss`, `val_loss`, `val_accuracy`, and the main training arguments.



- use `best_model.pt` when you want the best validation model
- use `last_model.pt` when you want the exact final epoch model

`metrics.json` stores the training arguments, per-epoch losses, validation
accuracy, and the best validation accuracy observed during training.
