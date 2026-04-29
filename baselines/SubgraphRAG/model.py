import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class SubgraphRAGModel(nn.Module):
    """
    Lightweight MLP triple-scorer with directional structural distance encoding,
    followed by an LLM that reasons over the top-scored triples.
    """

    def __init__(self, triple_dim=768, hidden_dim=256, llm_name_or_path=None):
        super().__init__()
        self.triple_scorer = nn.Sequential(
            nn.Linear(triple_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if llm_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        else:
            self.tokenizer = self.llm = None

    def score_triples(self, triple_embs):
        # triple_embs: (B, T, triple_dim)
        return self.triple_scorer(triple_embs).squeeze(-1)  # (B, T)

    @torch.no_grad()
    def generate_answer(self, prompts, max_new_tokens=128):
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
