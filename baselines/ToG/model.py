import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ToGModel(nn.Module):
    """
    Think-on-Graph: LLM agent scores candidate KG paths via beam search.
    Pre-explored beam_paths from the dataset are re-ranked and the best
    path is used to prompt the LLM for the final answer.
    """

    def __init__(self, model_name_or_path, beam_width=3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.beam_width = beam_width

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def score_paths(self, question, paths, max_new_tokens=8):
        # Score each path by asking the LLM if it is relevant to the question.
        prompts = [
            f'Question: {question}\nIs this reasoning path relevant? Path: {p}\nAnswer yes or no:'
            for p in paths
        ]
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        out = self.llm(**inputs)
        # use log-prob of first generated token as relevance score
        log_probs = out.logits[:, -1, :].log_softmax(-1)
        yes_id = self.tokenizer.encode('yes', add_special_tokens=False)[0]
        return log_probs[:, yes_id].tolist()

    @torch.no_grad()
    def generate_answer(self, prompts, max_new_tokens=128):
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
