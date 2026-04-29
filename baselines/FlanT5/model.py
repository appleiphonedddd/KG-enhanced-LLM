import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer


class FlanT5Model(nn.Module):
    def __init__(self, model_name_or_path='google/flan-t5-xl'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=64):
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.t5.device)
        ids = self.t5.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
