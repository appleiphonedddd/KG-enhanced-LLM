import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TransferNetModel(nn.Module):
    """
    Multi-step reasoning: attend to question tokens per hop to score relations,
    then transfer entity scores along those activated relations.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased',
                 num_relations=5000, num_hops=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.num_hops = num_hops
        # one relation-attention projection per hop
        self.hop_proj = nn.ModuleList(
            [nn.Linear(hidden, num_relations) for _ in range(num_hops)]
        )

    def forward(self, input_ids, attention_mask, entity_scores, rel_adj):
        # entity_scores: (B, E)  initial topic-entity indicator
        # rel_adj: (B, R, E, E) per-relation adjacency (row = source, col = dest)
        token_embs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                         # (B, L, d)
        s = entity_scores
        for hop in range(self.num_hops):
            r_w = F.softmax(self.hop_proj[hop](token_embs.mean(1)), dim=-1)  # (B, R)
            # weighted sum of per-relation adjacency matrices
            adj = (r_w[:, :, None, None] * rel_adj).sum(1)    # (B, E, E)
            s = torch.bmm(adj.transpose(1, 2), s.unsqueeze(-1)).squeeze(-1)
            s = F.softmax(s, dim=-1)
        return s                                    # (B, E)
