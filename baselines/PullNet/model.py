import torch
import torch.nn as nn
from transformers import AutoModel


class PullNetModel(nn.Module):
    """
    Two-stage GCN: a retrieval GCN expands the subgraph iteratively,
    then an extraction GCN scores answer entities.
    Both stages share a BERT question encoder.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased',
                 num_gcn_layers=2, num_iter=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.num_iter = num_iter
        self.retrieval_gcn = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_gcn_layers)]
        )
        self.extract_gcn = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_gcn_layers)]
        )
        self.out = nn.Linear(hidden, 1)

    def _gcn(self, layers, h, adj):
        for layer in layers:
            h = torch.relu(layer(torch.bmm(adj, h)))
        return h

    def forward(self, input_ids, attention_mask, entity_embs, adj):
        # entity_embs: (B, E, d)  adj: (B, E, E)
        q = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        h = entity_embs + q.unsqueeze(1)
        for _ in range(self.num_iter):
            h = self._gcn(self.retrieval_gcn, h, adj)
        h = self._gcn(self.extract_gcn, h, adj)
        return self.out(h).squeeze(-1)              # (B, E)
