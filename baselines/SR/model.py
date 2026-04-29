import torch
import torch.nn as nn
from transformers import AutoModel


class SRModel(nn.Module):
    """
    Trainable subgraph retriever decoupled from downstream reasoning.
    Scores (question, relation) pairs via a bilinear interaction;
    top-scoring triples form the retrieved subgraph.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased',
                 rel_vocab_size=6000, rel_dim=300):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.rel_embed = nn.Embedding(rel_vocab_size, rel_dim)
        self.scorer = nn.Bilinear(self.encoder.config.hidden_size, rel_dim, 1)

    def forward(self, input_ids, attention_mask, relation_ids):
        # relation_ids: (B, T) candidate relation ids
        q = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]                   # (B, H)
        r = self.rel_embed(relation_ids)             # (B, T, rel_dim)
        q_exp = q.unsqueeze(1).expand(-1, r.size(1), -1)
        return self.scorer(q_exp, r).squeeze(-1)    # (B, T)
