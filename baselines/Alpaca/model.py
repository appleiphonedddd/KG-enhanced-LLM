import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class AlpacaModel(nn.Module):
    def __init__(self, model_name_or_path='tatsu-lab/alpaca-7b'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=128):
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
