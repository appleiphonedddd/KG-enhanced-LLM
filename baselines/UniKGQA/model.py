import torch
import torch.nn as nn
from transformers import AutoModel


class UniKGQAModel(nn.Module):
    """
    Unified retrieval and reasoning with shared BERT encoder.
    Retrieval head scores (question, subgraph-tuple) pairs;
    reasoning head scores answer-entity candidates given the retrieved subgraph.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.retrieval_head = nn.Linear(hidden, 1)
        self.reasoning_head = nn.Linear(hidden, 1)

    def _cls(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]

    def retrieve(self, input_ids, attention_mask):
        # input: (question, tuple) pairs encoded together — (B*T, L)
        return self.retrieval_head(self._cls(input_ids, attention_mask)).squeeze(-1)

    def reason(self, input_ids, attention_mask):
        # input: (question + retrieved subgraph, entity) pairs — (B*E, L)
        return self.reasoning_head(self._cls(input_ids, attention_mask)).squeeze(-1)
