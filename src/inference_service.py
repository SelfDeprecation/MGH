import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABELS = ["O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND", "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"]

class InferenceService:
    def __init__(self, model_path="model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)

    def predict(self, text: str):
        tokens = text.split()
        inputs = self.tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True).to(self.device)
        outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
        word_ids = inputs.word_ids()
        labels = []
        for word_idx in range(len(tokens)):
            token_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
            if token_indices:
                labels.append(LABELS[predictions[token_indices[0]]])
            else:
                labels.append("O")
        return list(zip(tokens, labels))
