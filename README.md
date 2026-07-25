# PARS

This repository releases the code for the PARS workflow described in the paper. If you use PARS in your work, please cite the following paper:

Yiheng Tao, Yihe Zhang, Matthew Dearing, Xin Wang, Yuping Fan, Michael Papka, Zhiling Lan, “Ranking Before Serving: Low-Latency LLM Serving via Pairwise Learning-to-Rank”, Proc. of ISC High Performance 2026.

PARS is a prompt-aware scheduling approach designed to approximate
shortest-job-first style decisions for LLM serving. The goal is to reduce
end-to-end latency by using a lightweight proxy signal before requests are sent
to LLMs, substantially mitigating head-of-line blocking, improving
user experience, and reducing serving cost.

This release is organized around three stages:

- `data_preprocess/`: dataset download, filtering, train/val split, and pairwise sample generation
- `predictor_train/`: BERT-based pairwise ranker training
- `predictor_serving/`: FastAPI service for single and batch predictor inference

## What is included

- End-to-end preprocessing scripts for four GPT4-based datasets:
  `alpaca`, `code`, `lmsys`, and `math`
- Pairwise predictor training code
- Predictor serving code with both single-request and batch inference APIs

## Repository layout

The repository is organized as a simple three-stage pipeline: first prepare
pairwise supervision data, then train the predictor, and finally serve the
trained model for online scoring or scheduler-side integration.

`data_preprocess/`

- One script per dataset
- Downloads the source dataset
- Produces `train_data.json`, `val_data.json`, and pairwise train/val files

`predictor_train/`

- Trains a BERT-based pairwise ranking model from pairwise JSON data
- Saves `best_model.pt`, `last_model.pt`, and `metrics.json`

`predictor_serving/`

- Loads a trained checkpoint
- Exposes `/score`, `/score_batch`, `/compare`, and `/compare_batch`
- Supports batch tokenization and batch forward inference

## End-to-end workflow

### 1. Data preprocessing

Install preprocessing dependencies:

```bash
cd data_preprocess
pip install -r requirements.txt
```

Generate pairwise data for one dataset:

```bash
python scripts/preprocess_alpaca_gpt4.py --output-dir outputs/alpaca/gpt4
python scripts/preprocess_code_gpt4.py --output-dir outputs/code/gpt4
python scripts/preprocess_lmsys_gpt4.py --output-dir outputs/lmsys/gpt4
python scripts/preprocess_math_gpt4.py --output-dir outputs/math/gpt4
```

By default, all four preprocessing scripts generate pairwise files with
`threshold=0.2`.
Here `threshold` is the minimum relative response-length difference required
for constructing a pairwise sample, so that the training pairs reflect a clear
ordering signal instead of nearly indistinguishable prompt pairs.

### 2. Predictor training

Install training dependencies:

```bash
cd ../predictor_train
pip install -r requirements.txt
```

Example training command:

```bash
python scripts/train_pairwise_bert.py \
  --train-file ../data_preprocess/outputs/alpaca/gpt4/train_pairs_length_diff_0.2.json \
  --val-file ../data_preprocess/outputs/alpaca/gpt4/val_pairs_length_diff_0.2.json \
  --output-dir outputs/alpaca_gpt4_bert
```

The released training script defaults to:

- `bert-base-uncased`
- `num_epochs=3`


### 3. Predictor serving

Install serving dependencies:

```bash
cd ../predictor_serving
pip install -r requirements.txt
```

Launch the service:

```bash
PREDICTOR_MODEL_PATH=../predictor_train/outputs/alpaca_gpt4_bert/best_model.pt \
uvicorn scripts.serve_predictor_score:app --host 0.0.0.0 --port 8000
```

Quick test:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain machine learning simply."}'
```

## Dataset note

The preprocessing scripts in this release target GPT4-based data preparation
paths. If you want to reproduce LLaMA-, Deepseek-R1-, or other model-generated variants,
please generate those datasets manually and then reuse the same training and
serving components.

If you want to integrate PARS into vLLM or another inference platform, use
`predictor_serving/` to obtain prompt scores, assign request priorities through
the platform scheduler (for example, vLLM's priority scheduler), and use the
official benchmark code of vLLM or the target platform to simulate request
arrivals and execution.

Our paper will appear at `ISC High Performance 2026`, `June 22--26, 2026`, in
`Hamburg, Germany`.
