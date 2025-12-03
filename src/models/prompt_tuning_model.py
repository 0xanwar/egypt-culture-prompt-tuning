import torch
import torch.nn as nn
from transformers import BertModel
from src.config import ModelConfig


class CulturalPromptTuning(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, config.prompt_len, self.bert.config.hidden_size)
        )
        nn.init.normal_(self.prompt_embeddings, std=0.02)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        token_embeds = self.bert.embeddings(input_ids)
        batch_size = input_ids.shape[0]
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)
        combined_embeds = torch.cat(
            [
                token_embeds[:, :1, :],  # [CLS]
                prompts,
                token_embeds[:, 1:, :],
            ],
            dim=1,
        )

        prompt_mask = torch.ones(batch_size, self.config.prompt_len).to(
            attention_mask.device
        )
        combined_mask = torch.cat(
            [attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]], dim=1
        )

        combined_embeds = combined_embeds[:, :512, :]
        combined_mask = combined_mask[:, :512]

        outputs = self.bert(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        cls_repr = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_repr)
