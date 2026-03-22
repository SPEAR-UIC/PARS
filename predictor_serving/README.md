# Predictor Serving

This directory contains the serving code for the trained pairwise prompt ranker.

The service loads a trained checkpoint from `predictor_train/` and exposes HTTP
endpoints for prompt scoring and prompt comparison.

The released version supports batch inference. Single-request endpoints reuse
the same batch scoring path internally, and the batch endpoints run batched
tokenization plus batched forward inference for better throughput.

## Install

```bash
pip install -r requirements.txt
```

## Model checkpoint

Typical checkpoint paths from `../predictor_train/`:

- `../predictor_train/outputs/alpaca_gpt4_bert/best_model.pt`
- `../predictor_train/outputs/alpaca_gpt4_bert/last_model.pt`
- `../predictor_train/outputs/lmsys_gpt4_bert/best_model.pt`

Use `best_model.pt` if you want the checkpoint with the best validation
accuracy. Use `last_model.pt` if you want the final epoch checkpoint.

## Run

```bash
uvicorn scripts.serve_predictor_score:app --host 0.0.0.0 --port 8000
```

You can override model settings with environment variables:

- `PREDICTOR_MODEL_PATH`: checkpoint path
- `PREDICTOR_MODEL_NAME`: encoder name, default `bert-base-uncased`
- `PREDICTOR_MAX_LENGTH`: tokenizer max length, default `128`
- `PREDICTOR_DEVICE`: force device such as `cpu` or `cuda`

Example:

```bash
PREDICTOR_MODEL_PATH=../predictor_train/outputs/alpaca_gpt4_bert/best_model.pt \
PREDICTOR_MODEL_NAME=bert-base-uncased \
uvicorn scripts.serve_predictor_score:app --host 0.0.0.0 --port 8000
```

## Endpoints

`GET /`

- health check:

```bash
curl http://127.0.0.1:8000/
```

`POST /score`

- one-command example:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain machine learning simply."}'
```

- response body:

```json
{
  "prompt": "Explain machine learning simply.",
  "score": 0.42
}
```

A larger score means the predictor estimates that the prompt is more likely to
produce a longer response.

`POST /score_batch`

- one-command example:

```bash
curl -X POST http://127.0.0.1:8000/score_batch \
  -H "Content-Type: application/json" \
  -d '{"prompts":["Explain machine learning simply.","Give a detailed explanation of machine learning."]}'
```

- response body:

```json
{
  "prompts": [
    "Explain machine learning simply.",
    "Give a detailed explanation of machine learning."
  ],
  "scores": [0.18, 0.73]
}
```

`POST /compare`

- one-command example:

```bash
curl -X POST http://127.0.0.1:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt_a":"Explain machine learning simply.","prompt_b":"Give a detailed explanation of machine learning."}'
```

- response body:

```json
{
  "prompt_a": "Explain machine learning simply.",
  "prompt_b": "Give a detailed explanation of machine learning.",
  "score_a": 0.18,
  "score_b": 0.73,
  "winner": "prompt_b"
}
```

`POST /compare_batch`

- one-command example:

```bash
curl -X POST http://127.0.0.1:8000/compare_batch \
  -H "Content-Type: application/json" \
  -d '{"pairs":[{"prompt_a":"Explain machine learning simply.","prompt_b":"Give a detailed explanation of machine learning."},{"prompt_a":"Summarize AI.","prompt_b":"Write a long tutorial on artificial intelligence."}]}'
```

- response body:

```json
{
  "results": [
    {
      "prompt_a": "Explain machine learning simply.",
      "prompt_b": "Give a detailed explanation of machine learning.",
      "score_a": 0.18,
      "score_b": 0.73,
      "winner": "prompt_b"
    },
    {
      "prompt_a": "Summarize AI.",
      "prompt_b": "Write a long tutorial on artificial intelligence.",
      "score_a": 0.11,
      "score_b": 0.84,
      "winner": "prompt_b"
    }
  ]
}
```
