import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification


def predict_sentence(sentence: str,
                     model: AutoModelForTokenClassification,
                     tokenizer: AutoTokenizer,
                     device: torch.device,
                     id2label: dict):
    """
    Делает предсказание для одной строки.
    Возвращает список кортежей (start_index, end_index, label).
    """
    # 1) ищем слова в строке
    matches = list(re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+", sentence))
    if not matches:
        return []

    tokens = [m.group(0) for m in matches]
    positions = [(m.start(), m.end()) for m in matches]

    # 2) токенизация с привязкой к словам
    tokenized = tokenizer(tokens,
                          is_split_into_words=True,
                          return_tensors="pt",
                          truncation=True)

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 3) предсказание модели
    with torch.no_grad():
        if attention_mask is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

    # 4) маппинг токенов к словам
    word_ids = tokenized.word_ids(batch_index=0)

    annotation = []
    for word_idx, (start_char, end_char) in enumerate(positions):
        space_pos = sentence.find(" ", end_char)
        if space_pos != -1:
            end_index = space_pos
        else:
            end_index = end_char

        token_indices = [i for i, w in enumerate(word_ids) if w == word_idx]
        if token_indices:
            pred_label_idx = preds[token_indices[0]]
            label = id2label[pred_label_idx]
        else:
            label = "O"

        annotation.append((start_char, end_index, label))

    return annotation


def run_inference(model_path: str = "model",
                  submission_path: str = "data/submission.csv",
                  output_path: str = "data/submission_with_preds.csv"):
    """
    Загружает модель/токенайзер, прогоняет все строки из submission.csv (колонка 'sample')
    и сохраняет файл с колонками sample, annotation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # загрузка модели и токенайзера
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    # достаем словари меток из модели
    id2label = model.config.id2label

    # читаем submission
    try:
        df = pd.read_csv(submission_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(submission_path, sep=";", engine="python")

    if "sample" not in df.columns:
        raise ValueError("В submission.csv ожидается колонка 'sample'.")

    preds = []
    for sample in df["sample"].astype(str).tolist():
        ann = predict_sentence(sample, model, tokenizer, device, id2label)
        preds.append(str(ann))

    df["annotation"] = preds
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    run_inference()
