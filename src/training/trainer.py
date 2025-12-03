import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.config import TrainingConfig, ModelConfig
from src.models.prompt_tuning_model import CulturalPromptTuning


class CulturalTrainer:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device(training_config.device)
        self.model = CulturalPromptTuning(model_config).to(self.device)
        self.optimizer = self._create_optimizer()
        self.criterion = torch.nn.CrossEntropyLoss()

    def _create_optimizer(self):
        return AdamW(
            [
                {
                    "params": self.model.prompt_embeddings,
                    "lr": self.training_config.learning_rate_prompts,
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.training_config.learning_rate_classifier,
                },
            ]
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Starting training for {self.training_config.num_epochs} epochs...")

        for epoch in range(self.training_config.num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            val_acc = self._validate(val_loader)
            print(
                f"Epoch {epoch + 1}/{self.training_config.num_epochs} | "
                f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Save model
        torch.save(self.model.state_dict(), self.training_config.model_save_path)
        print(f"✅ Model saved to {self.training_config.model_save_path}")
        return self.model

    def _validate(self, val_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0
