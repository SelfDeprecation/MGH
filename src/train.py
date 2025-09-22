import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import torch
from seqeval.metrics import classification_report, f1_score
from .preprocess import load_csv, dataset_to_token_labels

LABELS = ["O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND", "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"]

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABELS.index(label[word_idx]))
            else:
                label_ids.append(LABELS.index(label[word_idx]) if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_labels = [[LABELS[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABELS[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    return {"f1": f1_score(true_labels, true_predictions), "report": classification_report(true_labels, true_predictions)}

def main(args):
    df = load_csv(args.data_path)
    items = dataset_to_token_labels(df)
    dataset = Dataset.from_list(items)
    ds = DatasetDict({"train": dataset})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = ds["train"].map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(LABELS))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=100,
        save_total_limit=2,
        metric_for_best_model="f1",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # run prediction on submission.csv
    from predict import run_inference
    run_inference(model_path=args.output_dir, submission_path="data/submission.csv", output_path="data/submission_with_preds.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--model_name", type=str, default="DeepPavlov/rubert-base-cased")
    args = parser.parse_args()
    main(args)
