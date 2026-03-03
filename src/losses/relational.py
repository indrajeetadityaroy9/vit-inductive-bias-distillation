import torch
import torch.nn.functional as F


def geometric_relational_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_attn: torch.Tensor,
    *,
    has_cls_token: bool,
) -> torch.Tensor:
    """Attention-weighted Procrustes loss.

    When has_cls_token=True, CLS-token attention weights encode token
    importance. When False, mean attention across query dimension serves
    as importance proxy.
    """
    s = student_tokens.float()
    t = teacher_tokens.float()
    N_s = s.shape[1]

    if has_cls_token:
        # Attention is [B, H, N+1, N+1]; CLS at index 0
        w = teacher_attn[:, :, 0, 1:].mean(dim=1)
    else:
        # No CLS: mean attention across query dimension
        w = teacher_attn.mean(dim=(1, 2))

    if w.shape[1] != N_s:
        w = F.interpolate(
            w.unsqueeze(1), size=N_s, mode="linear", align_corners=False,
        ).squeeze(1)

    w = w / w.sum(dim=-1, keepdim=True)

    mu_s = (w.unsqueeze(-1) * s).sum(dim=1, keepdim=True)
    mu_t = (w.unsqueeze(-1) * t).sum(dim=1, keepdim=True)
    s_c = s - mu_s
    t_c = t - mu_t

    w_sqrt = w.unsqueeze(-1).sqrt()
    s_w = w_sqrt * s_c
    t_w = w_sqrt * t_c

    tr_s = (s_w * s_w).sum(dim=(1, 2))
    tr_t = (t_w * t_w).sum(dim=(1, 2))
    cross = torch.bmm(s_w.transpose(1, 2), t_w)
    nuclear = torch.linalg.svdvals(cross).sum(dim=-1)

    return (tr_s + tr_t - 2.0 * nuclear).mean()
