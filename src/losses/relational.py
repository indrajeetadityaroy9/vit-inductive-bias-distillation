import torch
import torch.nn.functional as F


def geometric_relational_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_attn: torch.Tensor,
) -> torch.Tensor:
    """Attention-weighted Procrustes loss.

    CLS-token attention weights encode token importance (arXiv 2509.25253, Theorem 3).
    Teacher tokens remain attached to preserve projector gradients.
    """
    s = student_tokens.float()
    t = teacher_tokens.float()
    N_s = s.shape[1]

    # CLS-to-patch attention → per-token importance weights.
    w = teacher_attn[:, :, 0, 1:].mean(dim=1)  # [B, N_teacher]

    # Align to student spatial grid when teacher has different token count.
    if w.shape[1] != N_s:
        w = F.interpolate(
            w.unsqueeze(1), size=N_s, mode="linear", align_corners=False,
        ).squeeze(1)

    w = w / w.sum(dim=-1, keepdim=True)
    s = w.unsqueeze(-1) * s
    t = w.unsqueeze(-1) * t

    s_c = s - s.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)

    tr_s = (s_c * s_c).sum(dim=(1, 2))
    tr_t = (t_c * t_c).sum(dim=(1, 2))
    cross = torch.bmm(s_c.transpose(1, 2), t_c)
    nuclear = torch.linalg.svdvals(cross).sum(dim=-1)

    return (tr_s + tr_t - 2.0 * nuclear).mean()
