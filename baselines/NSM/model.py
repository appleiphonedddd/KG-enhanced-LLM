import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class NSMModel(nn.Module):
    """
    Teacher-student framework with GNN reasoning over a KG subgraph.
    The student propagates entity states; the teacher supplies hop-level targets.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased', num_hops=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.num_hops = num_hops
        self.node_update = nn.GRUCell(hidden, hidden)
        self.hop_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(num_hops)]
        )

    def forward(self, input_ids, attention_mask, entity_embs, adj):
        # entity_embs: (B, E, d)  adj: (B, E, E) normalised adjacency
        q = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]                   # (B, d)
        B, E, d = entity_embs.shape
        h = (entity_embs + q.unsqueeze(1)).view(B * E, d)
        hop_scores = []
        for hop in range(self.num_hops):
            msg = torch.bmm(adj, h.view(B, E, d)).view(B * E, d)
            h = self.node_update(msg, h)
            scores = self.hop_heads[hop](h.view(B, E, d)).squeeze(-1)  # (B, E)
            hop_scores.append(scores)
        return hop_scores                           # list of (B, E) per hop
