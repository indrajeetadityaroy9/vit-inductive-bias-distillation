import torch


def bures_wasserstein_loss(
    student_tokens: torch.Tensor, teacher_tokens: torch.Tensor, *, eps: float = 1e-5,
) -> torch.Tensor:
    """Bures-Wasserstein loss on token Gaussian statistics.

    Uses diagonal covariance when N < D, full covariance otherwise.
    """
    s = student_tokens.float()
    t = teacher_tokens.float()

    B, N, D = s.shape

    mu_s, mu_t = s.mean(1), t.mean(1)
    mean_loss = (mu_s - mu_t).pow(2).sum(-1).mean()

    s_c = s - mu_s.unsqueeze(1)
    t_c = t - mu_t.unsqueeze(1)

    if N < D:
        var_s = s_c.pow(2).mean(dim=1) + eps
        var_t = t_c.pow(2).mean(dim=1) + eps
        cov_loss = (var_s.sqrt() - var_t.sqrt()).pow(2).sum(-1).mean()
    else:
        eps_eye = eps * torch.eye(D, device=s.device, dtype=s.dtype).unsqueeze(0)
        cov_s = torch.bmm(s_c.transpose(1, 2), s_c) / N + eps_eye
        cov_t = torch.bmm(t_c.transpose(1, 2), t_c) / N + eps_eye

        L_s = torch.linalg.cholesky(cov_s)
        L_t = torch.linalg.cholesky(cov_t)

        nuclear = torch.linalg.svdvals(
            torch.bmm(L_t.transpose(1, 2), L_s)
        ).sum(dim=-1)
        trace_s = torch.diagonal(cov_s, dim1=-2, dim2=-1).sum(-1)
        trace_t = torch.diagonal(cov_t, dim1=-2, dim2=-1).sum(-1)
        cov_loss = (trace_s + trace_t - 2 * nuclear).mean()

    return mean_loss + cov_loss
