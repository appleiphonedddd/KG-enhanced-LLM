import torch

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tp = torch.sum((pred == 1) & (target == 1)).item()
        fp = torch.sum((pred == 1) & (target == 0)).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        return precision


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tp = torch.sum((pred == 1) & (target == 1)).item()
        fp = torch.sum((pred == 1) & (target == 0)).item()
        recall = tp / (tp + fp) if (tp + fp) > 0 else 0
        return recall


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tp = torch.sum((pred == 1) & (target == 1)).item()
        fp = torch.sum((pred == 1) & (target == 0)).item()
        fn = torch.sum((pred == 0) & (target == 1)).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def hitsat1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = torch.sum(pred == target).item()
        return correct / len(target)