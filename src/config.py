from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    prompt_len: int = 10
    num_labels: int = 5
    max_length: int = 128

@dataclass
class TrainingConfig:
    num_epochs: int = 15
    batch_size: int = 8
    learning_rate_prompts: float = 5e-4
    learning_rate_classifier: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path: str = "models/final_model_no_early_stop.pt"

CULTURAL_THEMES = {
    "religious_celebration": 0,
    "family_and_respect": 1,
    "national_pride": 2,
    "cultural_heritage": 3,
    "community_generosity": 4
}