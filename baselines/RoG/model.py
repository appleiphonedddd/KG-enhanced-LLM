import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class RoGPlanner(nn.Module):
    """Fine-tuned LLM for both planning (relation paths) and reasoning (final answer)."""

    def __init__(self, model_name_or_path):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate_paths(self, questions, num_paths=5, max_new_tokens=64):
        inputs = self.tokenizer(
            questions, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        ids = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_paths,
            num_return_sequences=num_paths,
        )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_answer(self, prompts, max_new_tokens=128):
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)


def retrieve_paths(topic_entity, relation_paths, kg):
    """
    Walk the KG from topic_entity following each relation path.

    Args:
        topic_entity: starting entity MID string
        relation_paths: list of relation sequences, e.g. [['rel1', 'rel2'], ...]
        kg: dict mapping entity MID -> list of (relation, neighbor_MID) tuples

    Returns:
        list of entity-relation chains, e.g. [['e0', 'r1', 'e1', 'r2', 'e2'], ...]
    """
    results = []
    for rel_path in relation_paths:
        results.extend(_walk(kg, [[topic_entity]], rel_path))
    return results


def _walk(kg, frontier, relations):
    for relation in relations:
        next_frontier = []
        for path in frontier:
            for rel, neighbor in kg.get(path[-1], []):
                if rel == relation:
                    next_frontier.append(path + [rel, neighbor])
        frontier = next_frontier
    return frontier
