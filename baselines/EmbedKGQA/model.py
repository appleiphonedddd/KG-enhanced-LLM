import torch
import torch.nn as nn
from transformers import AutoModel


class EmbedKGQAModel(nn.Module):
    """
    Projects question into KG embedding space and ranks answer candidates
    by inner product with pre-trained entity embeddings.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased', entity_dim=400):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, entity_dim)

    def forward(self, input_ids, attention_mask, entity_embeddings):
        # entity_embeddings: (B, num_candidates, entity_dim)
        cls = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        q_emb = self.proj(cls)                                          # (B, entity_dim)
        scores = (q_emb.unsqueeze(1) * entity_embeddings).sum(-1)      # (B, num_candidates)
        return scores
