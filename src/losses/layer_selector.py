import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _retract_to_stiefel(linear: nn.Linear) -> None:
    U, _, Vt = torch.linalg.svd(linear.weight, full_matrices=False)
    linear.weight.copy_(U @ Vt)


@torch.no_grad()
def marchenko_pastur_rank(features: torch.Tensor) -> int:
    M, D = features.shape
    q = D / M
    if M >= D:
        cov = features.T @ features / M
    else:
        cov = features @ features.T / M
    eigvals = torch.linalg.eigvalsh(cov)
    sigma2 = eigvals.median().item()
    lambda_plus = sigma2 * (1 + q ** 0.5) ** 2
    rank = (eigvals > lambda_plus).sum().item()
    return int(rank)


def _grassmann_subspace(
    z_flat: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    z = z_flat.float()
    z = z - z.mean(dim=0, keepdim=True)
    M, d = z.shape
    cov = z.T @ z / M
    cov = cov + 1e-4 * torch.eye(d, device=cov.device, dtype=cov.dtype)
    _, eigvecs = torch.linalg.eigh(cov)
    return eigvecs[:, -k:]


class GrassmannianLayerSelector(nn.Module):
    def __init__(
        self,
        num_extraction_points: int,
        student_dim: int,
        teacher_dim: int,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.subspace_rank = student_dim // 4
        self._rank_calibrated = False

        proj_s = torch.empty(student_dim, student_dim)
        proj_t = torch.empty(student_dim, teacher_dim)
        nn.init.orthogonal_(proj_s)
        nn.init.orthogonal_(proj_t)
        self.register_buffer("proj_s", proj_s)
        self.register_buffer("proj_t", proj_t)

        self.log_temperatures = nn.Parameter(
            torch.full(
                (num_extraction_points,),
                math.log(math.exp(1.0) - 1),
            )
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.log_temperatures)

    def _calibrate_rank(self, sample_tokens: torch.Tensor) -> None:
        if self._rank_calibrated:
            return
        with torch.no_grad():
            sample_z = sample_tokens.reshape(-1, sample_tokens.shape[2]) @ self.proj_t.T
            auto_rank = marchenko_pastur_rank(sample_z)
            self.subspace_rank = min(auto_rank, self.student_dim - 1)
        self._rank_calibrated = True

    def _compute_teacher_state(
        self,
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        teacher_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        D_t = all_teacher_tokens[teacher_indices[0]].shape[2]

        stacked_tokens = torch.stack([all_teacher_tokens[idx] for idx in teacher_indices])
        stacked_attns = torch.stack([all_teacher_attns[idx] for idx in teacher_indices])

        subspaces = {}
        with torch.no_grad():
            for idx in teacher_indices:
                z_t = all_teacher_tokens[idx].reshape(-1, D_t) @ self.proj_t.T
                subspaces[idx] = _grassmann_subspace(z_t, k=self.subspace_rank)

        return stacked_tokens, stacked_attns, subspaces

    def _mix_for_student_layer(
        self,
        i: int,
        s_tokens: torch.Tensor,
        teacher_indices: list[int],
        stacked_tokens: torch.Tensor,
        stacked_attns: torch.Tensor,
        subspaces: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.subspace_rank

        D_s = s_tokens.shape[2]
        s_flat = s_tokens.reshape(-1, D_s)
        z_s = s_flat @ self.proj_s.T
        U_s = _grassmann_subspace(z_s, k=k)

        d_grass_sq = torch.zeros(len(teacher_indices), device=stacked_tokens.device)
        for j, t_idx in enumerate(teacher_indices):
            U_t = subspaces[t_idx]
            sigma = torch.linalg.svdvals(U_s.T @ U_t)
            theta = torch.acos(sigma.clamp(max=1.0 - 1e-7))
            d_grass_sq[j] = theta.pow(2).sum()

        tau = self.temperatures[i]
        weights = F.softmax(-d_grass_sq / (k * tau), dim=0)

        weights_mix = weights.to(stacked_tokens.dtype)
        mixed = (weights_mix.view(-1, 1, 1, 1) * stacked_tokens).sum(dim=0)
        mixed_attn = (weights_mix.view(-1, 1, 1, 1, 1) * stacked_attns).sum(dim=0)

        return mixed, mixed_attn

    def forward(
        self,
        student_tokens_per_layer: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        extraction_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        teacher_indices = sorted(all_teacher_tokens.keys())

        self._calibrate_rank(all_teacher_tokens[teacher_indices[0]])
        stacked_tokens, stacked_attns, subspaces = self._compute_teacher_state(
            all_teacher_tokens, all_teacher_attns, teacher_indices,
        )

        mixed_teachers = {}
        mixed_attentions = {}

        for i, s_layer in enumerate(extraction_indices):
            mixed, attn = self._mix_for_student_layer(
                i, student_tokens_per_layer[s_layer],
                teacher_indices, stacked_tokens, stacked_attns, subspaces,
            )
            mixed_teachers[s_layer] = mixed
            mixed_attentions[s_layer] = attn

        return mixed_teachers, mixed_attentions
