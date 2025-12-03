import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.data.dataset import CulturalTextDataset
from src.training.trainer import CulturalTrainer


def main():
    # Load data
    ds = load_dataset("Anwar12/Atomic-EgMM")
    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()

    # Initialize
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Create datasets
    train_dataset = CulturalTextDataset(train_df, tokenizer, model_config.max_length)
    val_dataset = CulturalTextDataset(val_df, tokenizer, model_config.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train
    trainer = CulturalTrainer(model_config, training_config)
    model = trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
