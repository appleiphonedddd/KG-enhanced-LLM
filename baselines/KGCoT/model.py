import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class KGCoTModel(nn.Module):
    """
    KG-enhanced CoT: a lightweight BERT step-scorer validates each reasoning
    step against the KG; the LLM then generates the final answer from the
    verified step sequence.
    """

    def __init__(self, pretrained_model_name='bert-base-uncased',
                 llm_name_or_path=None):
        super().__init__()
        self.step_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.step_scorer = nn.Linear(self.step_encoder.config.hidden_size, 1)
        if llm_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        else:
            self.tokenizer = self.llm = None

    def score_steps(self, input_ids, attention_mask):
        # input: (question, step) pairs encoded together — (B*S, L)
        cls = self.step_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.step_scorer(cls).squeeze(-1)    # (B*S,)

    @torch.no_grad()
    def generate_answer(self, prompts, max_new_tokens=128):
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
