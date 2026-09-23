import torch
from torch.nn import functional as F


def statistics(prediction, target, epsilon=1e-6, mask=None):
    """Return differentiable sums: [token NMSE, token cosine, valid count]."""
    x, y = prediction.float(), target.float()
    error = (x - y).square().mean(-1)
    nmse = error / y.square().mean(-1).clamp_min(epsilon)
    cosine = F.cosine_similarity(x, y, dim=-1, eps=epsilon)
    mask = torch.ones_like(nmse) if mask is None else mask.float()
    return torch.stack([(nmse * mask).sum(), (cosine * mask).sum(), mask.sum()])


def loss_sum(stats, cosine_weight):
    return stats[0] + cosine_weight * (stats[2] - stats[1])


def metrics(stats):
    nmse, cosine, count = stats.double().cpu().tolist()
    if count <= 0:
        raise ValueError("No valid validation tokens")
    return {"nmse": nmse / count, "rrms": (nmse / count) ** .5,
            "cosine": cosine / count, "tokens": int(count)}


def kl_sum(student_logits, teacher_logits):
    teacher = F.log_softmax(teacher_logits.float(), dim=-1)
    student = F.log_softmax(student_logits.float(), dim=-1)
    return (teacher.exp() * (teacher - student)).sum()
