import typing
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from configs import config

LABELS = ["O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND", "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"]

def predict_sentence(sentence: str, model, tokenizer, device: str) -> typing.List[typing.Tuple[int, str]]:
    tokens = sentence.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True).to(device)
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()
    labels = []
    for word_idx in range(len(tokens)):
        token_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
        if token_indices:
            labels.append(LABELS[predictions[token_indices[0]]])
        else:
            labels.append("O")
    annotation = [(i, lab) for i, lab in enumerate(labels)]
    return annotation


def predict(sentence: str) -> typing.List[typing.Tuple[int, str]]:
    model_path = config.BASE_MODEL_PATH
    device = config.BASE_DEVICE
    return predict.predict_sentence(
        sentence=sentence,
        model=AutoModelForTokenClassification.from_pretrained(model_path).to(device),
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        device=device
        )


def run_inference(model_path="model", submission_path="data/submission.csv", output_path="data/submission_with_preds.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    df = pd.read_csv(submission_path)
    preds = []
    for sample in df["sample"].tolist():
        ann = predict_sentence(sample, model, tokenizer, device)
        preds.append(str(ann))
    df["annotation"] = preds
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    run_inference()
