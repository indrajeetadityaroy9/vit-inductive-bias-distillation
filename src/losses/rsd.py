import torch
import torch.nn as nn


class RedundancySuppressionLoss(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_layers: int,
        *,
        kappa: float,
    ):
        super().__init__()
        self.kappa = kappa
        self.teacher_dim = teacher_dim

        hidden_dim = max(student_dim, teacher_dim)
        self.aad_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim),
            )
            for _ in range(num_layers)
        ])

        self.student_bn = nn.ModuleList([
            nn.BatchNorm1d(teacher_dim, affine=False)
            for _ in range(num_layers)
        ])
        self.teacher_bn = nn.ModuleList([
            nn.BatchNorm1d(teacher_dim, affine=False)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        student_intermediates: dict[int, torch.Tensor],
        teacher_intermediates: dict[int, torch.Tensor],
        layer_indices: list[int],
    ) -> torch.Tensor:
        losses = []
        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]
            teacher_tokens = teacher_intermediates[layer_idx]

            B, N, D_s = student_tokens.shape
            D_t = self.teacher_dim
            M = B * N

            s_2d = self.aad_projectors[i](
                student_tokens.reshape(M, D_s)
            ).float()
            t_2d = teacher_tokens.reshape(M, D_t).float()

            s_norm = self.student_bn[i](s_2d)
            t_norm = self.teacher_bn[i](t_2d)

            cc = s_norm.T @ t_norm / M
            target = torch.eye(D_t, device=cc.device, dtype=cc.dtype)
            diff = (cc - target).pow(2)

            off_diag = ~torch.eye(D_t, dtype=torch.bool, device=cc.device)
            diff[off_diag] *= self.kappa

            losses.append(diff.sum())

        return torch.stack(losses).mean()
