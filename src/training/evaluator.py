import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from src.config import CULTURAL_THEMES


class CulturalEvaluator:
    def __init__(self, model, device, id_to_theme=None):
        self.model = model
        self.device = device
        self.id_to_theme = id_to_theme or {v: k for k, v in CULTURAL_THEMES.items()}

    def evaluate(self, loader: DataLoader, dataset_name: str):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if not all_labels:
            print(f"No examples in {dataset_name}")
            return {}

        accuracy = sum(a == b for a, b in zip(all_preds, all_labels)) / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        results = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "predictions": all_preds,
            "labels": all_labels,
        }

        self._print_results(results, dataset_name)
        return results

    def _print_results(self, results, dataset_name):
        print(f"\n{'=' * 50}")
        print(f"🎯 {dataset_name.upper()} RESULTS")
        print(f"{'=' * 50}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")

    def predict_top2(self, text: str, tokenizer, max_length: int = 128):
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]

        top_indices = np.argsort(probabilities)[::-1][:2]
        return [
            {"theme": self.id_to_theme[idx], "id": idx, "confidence": float(prob)}
            for idx, prob in zip(top_indices, probabilities[top_indices])
        ]
