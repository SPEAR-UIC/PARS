import os

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = os.getenv("PREDICTOR_MODEL_PATH", "../predictor_train/outputs/alpaca_gpt4_bert/best_model.pt")
MODEL_NAME = os.getenv("PREDICTOR_MODEL_NAME", "bert-base-uncased")
MAX_LENGTH = int(os.getenv("PREDICTOR_MAX_LENGTH", "128"))
DEVICE = os.getenv("PREDICTOR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        score_a = self.score(input_ids_a, attention_mask_a)
        score_b = self.score(input_ids_b, attention_mask_b)
        return score_a, score_b


class ScoreRequest(BaseModel):
    prompt: str


class ScoreBatchRequest(BaseModel):
    prompts: list[str]


class CompareRequest(BaseModel):
    prompt_a: str
    prompt_b: str


class CompareItem(BaseModel):
    prompt_a: str
    prompt_b: str


class CompareBatchRequest(BaseModel):
    pairs: list[CompareItem]


app = FastAPI(title="Prompt Score Predictor API", version="1.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = PairwiseRanker(MODEL_NAME).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def encode_prompts(prompts):
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )


def get_prompt_scores(prompts):
    if not prompts:
        return []
    with torch.no_grad():
        encoding = encode_prompts(prompts)
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        score = model.score(input_ids, attention_mask)
    return [float(value) for value in score.detach().cpu().tolist()]


@app.get("/")
def root():
    return {
        "message": "Predictor score model is running.",
        "model_path": MODEL_PATH,
        "model_name": MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/score")
def score_prompt(request: ScoreRequest):
    score = get_prompt_scores([request.prompt])[0]
    return {
        "prompt": request.prompt,
        "score": score,
    }


@app.post("/score_batch")
def score_prompt_batch(request: ScoreBatchRequest):
    scores = get_prompt_scores(request.prompts)
    return {
        "prompts": request.prompts,
        "scores": scores,
    }


@app.post("/compare")
def compare_prompts(request: CompareRequest):
    score_a, score_b = get_prompt_scores([request.prompt_a, request.prompt_b])
    winner = "prompt_a" if score_a > score_b else "prompt_b"
    return {
        "prompt_a": request.prompt_a,
        "prompt_b": request.prompt_b,
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
    }


@app.post("/compare_batch")
def compare_prompt_batch(request: CompareBatchRequest):
    prompts_a = [item.prompt_a for item in request.pairs]
    prompts_b = [item.prompt_b for item in request.pairs]
    scores_a = get_prompt_scores(prompts_a)
    scores_b = get_prompt_scores(prompts_b)

    results = []
    for item, score_a, score_b in zip(request.pairs, scores_a, scores_b):
        results.append(
            {
                "prompt_a": item.prompt_a,
                "prompt_b": item.prompt_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": "prompt_a" if score_a > score_b else "prompt_b",
            }
        )

    return {"results": results}
