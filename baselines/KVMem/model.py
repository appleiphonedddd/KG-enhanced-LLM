import torch
import torch.nn as nn
import torch.nn.functional as F


class KVMemModel(nn.Module):
    """
    Key-Value Memory Network with separate key/value encodings per hop.
    Keys encode relation paths; values encode entity representations.
    """

    def __init__(self, vocab_size, embed_dim=128, num_hops=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.num_hops = num_hops
        self.key_encoders = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_hops)]
        )
        self.val_encoders = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_hops)]
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, query_ids, key_ids, val_ids):
        # query_ids: (B, Lq)  key_ids: (B, M, Lk)  val_ids: (B, M, Lv)
        u = self.embed(query_ids).mean(1)           # (B, d)
        k = self.embed(key_ids).mean(2)             # (B, M, d)
        v = self.embed(val_ids).mean(2)             # (B, M, d)
        for hop in range(self.num_hops):
            k_h = self.key_encoders[hop](k)
            v_h = self.val_encoders[hop](v)
            attn = F.softmax((u.unsqueeze(1) * k_h).sum(-1), dim=-1)  # (B, M)
            u = u + (attn.unsqueeze(-1) * v_h).sum(1)
        return self.out(u)                          # (B, vocab_size)
