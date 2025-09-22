import pandas as pd
import ast

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')

def parse_annotation(ann_str):
    try:
        parsed = ast.literal_eval(ann_str)
        labels = [label for _, label in parsed]
        return labels
    except Exception:
        return []

def dataset_to_token_labels(df: pd.DataFrame, text_col="sample", ann_col="annotation"):
    items = []
    for _, row in df.iterrows():
        tokens = list(str(row[text_col]).split())
        labels = row[ann_col]
        if isinstance(labels, str):
            labels = parse_annotation(labels)
        if len(labels) != len(tokens):
            labels = ["O"] * len(tokens)
        items.append({"tokens": tokens, "labels": labels})
    return items
