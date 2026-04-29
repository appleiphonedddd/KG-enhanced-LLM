import torch
import torch.nn as nn


class GraftNetModel(nn.Module):
    """
    LSTM question encoder + multi-layer GCN over a question-specific subgraph.
    Entity scores are gated by the question representation at the final layer.
    """

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_gcn_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        gcn_dim = hidden_dim * 2
        self.gcn = nn.ModuleList(
            [nn.Linear(gcn_dim, gcn_dim) for _ in range(num_gcn_layers)]
        )
        self.out = nn.Linear(gcn_dim, 1)

    def forward(self, q_ids, entity_embs, adj):
        # q_ids: (B, Lq)  entity_embs: (B, E, gcn_dim)  adj: (B, E, E) normalised
        q_out, _ = self.lstm(self.embed(q_ids))
        q = q_out.mean(1)                           # (B, gcn_dim)
        h = entity_embs
        for layer in self.gcn:
            h = torch.relu(layer(torch.bmm(adj, h)))
        scores = self.out(h * q.unsqueeze(1)).squeeze(-1)   # (B, E)
        return scores
