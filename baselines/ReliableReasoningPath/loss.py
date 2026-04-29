from baselines.ReliableReasoningPath.model import _bidirectional_loss
import torch.nn.functional as F


def semantic_path_loss(log_probs_per_path):
    """Lr1 (eq.2): mean NLL over the shortest-path set Γ*."""
    return -sum(log_probs_per_path) / len(log_probs_per_path)


def structural_path_loss(P_fwd, P_bwd, entity_labels):
    """Lr2 (eq.8): forward/backward KL + JS divergence across directions."""
    return _bidirectional_loss(P_fwd, P_bwd, entity_labels)
