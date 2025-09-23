import os
import re
import ast
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from .predict import run_inference

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

# =========================
# Разбор аннотаций
# =========================
def parse_annotation(annotation_str):
    try:
        ann = ast.literal_eval(annotation_str)
        return [(int(s), int(e), str(l)) for s, e, l in ann]
    except Exception:
        return []

# =========================
# Токенизация с выравниванием лейблов
# =========================
def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128,
    )

    aligned_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                label_ids.append(label2id[labels[word_idx]])
            prev_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized

# =========================
# Загрузка датасета
# =========================
def load_dataset(path):
    df = pd.read_csv(path, sep=";")

    # Парсим аннотации
    df["parsed"] = df["annotation"].apply(parse_annotation)

    tokens_list, labels_list = [], []
    for _, row in df.iterrows():
        text = str(row["sample"])
        anns = row["parsed"]

        tokens = re.findall(r"\S+", text)
        labels = ["O"] * len(tokens)

        for (s, e, lab) in anns:
            substring = text[s:e]
            for i, tok in enumerate(tokens):
                if substring in tok:
                    labels[i] = lab

        tokens_list.append(tokens)
        labels_list.append(labels)

    df["tokens"] = tokens_list
    df["labels"] = labels_list

    dataset = Dataset.from_pandas(df[["tokens", "labels"]])
    return dataset

# =========================
# Основной процесс
# =========================
def main(args):
    # print("📂 Загружаем датасет...")
    # dataset = load_dataset(args.data_path)

    # # train/val split
    # train_test = dataset.train_test_split(test_size=0.1, seed=42)
    # ds = DatasetDict({
    #     "train": train_test["train"],
    #     "validation": train_test["test"]
    # })

    # # собираем словарь меток
    # unique_labels = set(l for sublist in dataset["labels"] for l in sublist)
    # label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
    # id2label = {i: label for label, i in label2id.items()}

    # print("📝 Метки:", label2id)

    # tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    # def preprocess(batch):
    #     return tokenize_and_align_labels(batch, tokenizer, label2id)

    # tokenized_ds = ds.map(preprocess, batched=True)

    # model = AutoModelForTokenClassification.from_pretrained(
    #     "DeepPavlov/rubert-base-cased",
    #     num_labels=len(label2id),
    #     id2label=id2label,
    #     label2id=label2id,
    # )

    # data_collator = DataCollatorForTokenClassification(tokenizer)

    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=50,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_ds["train"],
    #     eval_dataset=tokenized_ds["validation"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )

    # print("🚀 Начинаем обучение...")
    # trainer.train()

    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"✅ Модель сохранена в {args.output_dir}")

    run_inference()

# =========================
# Точка входа
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--output_dir", type=str, default="./model")
    args = parser.parse_args()
    main(args)
