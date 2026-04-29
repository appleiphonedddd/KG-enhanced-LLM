import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SemanticPathGenerator(nn.Module):
    """
    Fine-tuned LLM for semantic reasoning path generation (§IV-A).
    Trained to minimise KL divergence between the LLM path distribution
    and the posterior of reliable paths in the KG (approximated as NLL
    over the shortest-path set Γ*).
    """

    def __init__(self, model_name_or_path='meta-llama/Llama-2-7b-chat-hf'):
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
    def embed(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True
        ).to(self.llm.device)
        out = self.llm(**inputs, output_hidden_states=True)
        return out.hidden_states[-1][:, -1]         # last token hidden state


class StructuralPathGenerator(nn.Module):
    """
    Relation-embedding + bidirectional distribution learning module (§IV-B,
    Algorithm 1).

    Architecture:
      GloVe → LSTM → hidden state vq (question)
      Attention-based recurrent decoder produces reasoning instructions {ω^i}
      Entity embeddings initialised from neighbouring relation embeddings
      Per-hop: match vector m, neighbour aggregation, FFN update, softmax dist.
      Bidirectional training adds a backward pass sharing the same parameters.
    """

    def __init__(self, glove_dim=300, hidden_dim=256, rel_dim=300, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.lstm = nn.LSTM(glove_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.W1 = nn.Linear(rel_dim, rel_dim, bias=False)
        self.W2 = nn.Linear(rel_dim, hidden_dim, bias=False)
        # entity dim = rel_dim; concatenated with aggregated hidden → rel_dim
        self.ffn = nn.Sequential(
            nn.Linear(rel_dim + hidden_dim, rel_dim),
            nn.ReLU(),
        )
        self.W_dist = nn.Linear(rel_dim, 1, bias=False)

    def _init_entity_embs(self, rel_embs, adj_mask):
        # rel_embs: (B, E, R, rel_dim)  adj_mask: (B, E, R) float
        return torch.sigmoid(
            (adj_mask.unsqueeze(-1) * self.W1(rel_embs)).sum(2)  # (B, E, rel_dim)
        )

    def _one_pass(self, vq, rel_embs, adj_mask):
        """Run forward or backward entity-distribution pass.

        vq: (B, hidden_dim) question encoding
        Returns list of (B, E) distributions, one per hop, and final entity embs.
        """
        B, E, R, _ = rel_embs.shape
        ve = self._init_entity_embs(rel_embs, adj_mask)     # (B, E, rel_dim)
        P = torch.ones(B, E, device=ve.device) / E
        omega = vq.unsqueeze(1)                             # (B, 1, hidden_dim)
        dists = []
        for _ in range(self.num_hops):
            omega, _ = self.decoder(omega)                  # (B, 1, hidden_dim)
            omega_i = omega.squeeze(1)                      # (B, hidden_dim)
            # match vector: σ(ω^i ⊙ W2*vr)  eq.(4)
            wr = self.W2(rel_embs)                          # (B, E, R, hidden_dim)
            m = torch.sigmoid(omega_i[:, None, None, :] * wr)  # (B, E, R, hidden_dim)
            # aggregate neighbours weighted by previous-step distribution  eq.(5)
            ve_tilde = (adj_mask.unsqueeze(-1) * m).sum(2)  # (B, E, hidden_dim)
            # weight by P^{i-1}
            ve_tilde = (P.unsqueeze(-1) * ve_tilde)
            # entity embedding update  eq.(6)
            ve = self.ffn(torch.cat([ve, ve_tilde], dim=-1))
            # entity distribution  eq.(7)
            P = torch.softmax(self.W_dist(ve).squeeze(-1), dim=-1)
            dists.append(P)
        return dists, ve

    def forward(self, q_glove, rel_embs, adj_mask):
        # q_glove: (B, L, glove_dim)
        # rel_embs: (B, E, R, rel_dim)  pre-computed relation embeddings
        # adj_mask: (B, E, R) float — 1 where (entity, relation) exists in KG
        _, (h, _) = self.lstm(q_glove)
        vq = h.squeeze(0)                           # (B, hidden_dim)

        P_fwd, ve_fwd = self._one_pass(vq, rel_embs, adj_mask)

        # backward: reverse relation graph (transpose adj) with same params
        adj_mask_bwd = adj_mask.transpose(1, 2) if adj_mask.dim() == 3 else adj_mask
        rel_embs_bwd = rel_embs.transpose(1, 2) if rel_embs.dim() == 4 else rel_embs
        P_bwd, _ = self._one_pass(vq, rel_embs_bwd, adj_mask_bwd)

        return P_fwd, P_bwd, vq, ve_fwd


class RethinkingModule(nn.Module):
    """
    Scores and filters reasoning paths by combined semantic + structural
    cosine similarity (§IV-C, Algorithm 2).

    S(q, γ) = λ1 · S1 + λ2 · S2
    Paths with S < θ are discarded; survivors are sorted descending.
    """

    def __init__(self, lambda1=0.5, lambda2=0.5, theta=0.6):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.theta = theta

    def forward(self, vq_sem, vq_struct, path_sem_embs, path_struct_embs):
        # vq_sem:         (B, d_s)  question embedding from semantic LLM
        # vq_struct:      (B, d_t)  question encoding from structural LSTM
        # path_sem_embs:  (B, P, d_s) per-path LLM hidden states
        # path_struct_embs: (B, P, d_t) per-path averaged entity embeddings
        s1 = F.cosine_similarity(vq_sem.unsqueeze(1), path_sem_embs, dim=-1)     # (B, P)
        s2 = F.cosine_similarity(vq_struct.unsqueeze(1), path_struct_embs, dim=-1)  # (B, P)
        s = self.lambda1 * s1 + self.lambda2 * s2
        return s, s > self.theta                    # scores (B,P), keep-mask (B,P)


class RRPModel(nn.Module):
    """
    Full Reliable Reasoning Path model (Algorithm 2).

    Training:
      semantic_loss  = NLL over reliable paths (§IV-A, eq.2)
      structural_loss = Lr2 via bidirectional KL + JS divergence (§IV-B, eq.8)
    Inference:
      1. Generate semantic paths from fine-tuned LLM.
      2. Extract structural paths from entity distribution P_fwd.
      3. Rethink: score, filter (θ), sort by S descending.
      4. Feed ordered paths + question to LLM for final answer.
    """

    def __init__(self, llm_name_or_path,
                 glove_dim=300, hidden_dim=256, rel_dim=300, num_hops=2,
                 lambda1=0.5, lambda2=0.5, theta=0.6):
        super().__init__()
        self.semantic = SemanticPathGenerator(llm_name_or_path)
        self.structural = StructuralPathGenerator(glove_dim, hidden_dim, rel_dim, num_hops)
        self.rethinking = RethinkingModule(lambda1, lambda2, theta)

    def forward(self, input_ids, attention_mask, labels,
                q_glove, rel_embs, adj_mask, entity_labels):
        """
        input_ids / attention_mask / labels: for the semantic LLM path generation
        q_glove: (B, L, glove_dim)
        rel_embs: (B, E, R, rel_dim)
        adj_mask: (B, E, R)
        entity_labels: (B, E) float — 1 at correct answer entity for each hop
        """
        sem_loss, _ = self.semantic(input_ids, attention_mask, labels)
        P_fwd, P_bwd, vq, _ = self.structural(q_glove, rel_embs, adj_mask)
        struct_loss = _bidirectional_loss(P_fwd, P_bwd, entity_labels)
        return sem_loss + struct_loss

    @torch.no_grad()
    def predict(self, questions, q_glove, rel_embs, adj_mask,
                num_paths=5, max_new_tokens=128):
        sem_paths = self.semantic.generate_paths(questions, num_paths)
        P_fwd, P_bwd, vq_struct, ve = self.structural(q_glove, rel_embs, adj_mask)

        vq_sem = self.semantic.embed(questions)         # (B, d_s)
        # structural path embedding: average entity embs weighted by final dist
        path_struct = (P_fwd[-1].unsqueeze(-1) * ve).sum(1, keepdim=True)  # (B, 1, rel_dim)
        path_sem = vq_sem.unsqueeze(1)                  # (B, 1, d_s)

        scores, mask = self.rethinking(vq_sem, vq_struct, path_sem, path_struct)

        prompts = _build_prompts(questions, sem_paths, scores, mask)
        return self.semantic.generate_paths(prompts, num_paths=1,
                                            max_new_tokens=max_new_tokens)


# ── helpers ──────────────────────────────────────────────────────────────────

def _js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction='batchmean') \
         + 0.5 * F.kl_div(m.log(), q, reduction='batchmean')


def _bidirectional_loss(P_fwd, P_bwd, entity_labels):
    """Lr2 from eq.(8): KL for forward/backward + JS cross-direction per step."""
    n = len(P_fwd)
    loss = torch.tensor(0.0, device=P_fwd[0].device)
    for i, (pf, pb) in enumerate(zip(P_fwd, P_bwd)):
        loss = loss + F.kl_div(pf.log(), entity_labels, reduction='batchmean')
        loss = loss + F.kl_div(pb.log(), entity_labels, reduction='batchmean')
        if i < n - 1:
            loss = loss + _js_divergence(P_fwd[i], P_bwd[n - 1 - i])
    return loss


def _build_prompts(questions, sem_paths, scores, mask):
    B = len(questions)
    num_paths = scores.size(1)
    prompts = []
    for b in range(B):
        kept = [(sem_paths[b * num_paths + p], scores[b, p].item())
                for p in range(num_paths) if mask[b, p].item()]
        kept.sort(key=lambda x: -x[1])
        path_str = '\n'.join(f'- {path}' for path, _ in kept) or '(none)'
        prompts.append(
            f'Please use the reasoning paths provided below to answer the question.\n'
            f'The reasoning paths are listed in order of importance, with the first '
            f'being the most important.\nYour task is to derive the simplest possible '
            f'answer and return all potential answers as a list.\n\n'
            f'Reasoning Paths:\n{path_str}\n\nQuestion: {questions[b]}'
        )
    return prompts
