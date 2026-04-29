def hits_at_1(pred, golds):
    """pred: top-1 predicted entity string; golds: list of gold entity strings."""
    return float(pred in set(golds))


def f1_score(preds, golds):
    """Set-based F1 between predicted and gold entity sets."""
    pred_set, gold_set = set(preds), set(golds)
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
