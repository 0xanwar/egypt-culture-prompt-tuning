import ast
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from src.utils.theme_mapper import CulturalThemeMapper
from src.config import CULTURAL_THEMES


class CulturalTextDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_length: int = 128
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._create_examples()

    def _parse_field(self, field_str):
        """Safely parse stringified lists"""
        if pd.isna(field_str) or not isinstance(field_str, str):
            return []
        try:
            if "'" in field_str:
                field_str = field_str.replace("'", '"')
            return ast.literal_eval(field_str)
        except:
            return [field_str] if field_str else []

    def _create_examples(self):
        examples = []
        for _, row in self.df.iterrows():
            event = row["event"]
            if not isinstance(event, str) or len(event.strip()) == 0:
                continue

            intents = self._parse_field(row["xIntent"])
            for intent in intents:
                if isinstance(intent, str):
                    theme = CulturalThemeMapper.map_intent_to_theme(intent)
                    if theme in CULTURAL_THEMES:
                        examples.append((event, CULTURAL_THEMES[theme]))
                        break
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
