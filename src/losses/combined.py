import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.attention import AttentionDistillationLoss
from src.losses.layer_selector import (
    GrassmannianLayerSelector,
    _retract_to_stiefel,
)
from src.losses.relational import geometric_relational_loss
from src.losses.rsd import RedundancySuppressionLoss
from src.losses.spectral import bures_wasserstein_loss


class CrossAttentionProjector(nn.Module):
    def __init__(self, num_student_tokens: int, teacher_dim: int, proj_dim: int, *, num_heads: int = 4):
        super().__init__()
        self.teacher_proj = nn.Linear(teacher_dim, proj_dim)
        self.queries = nn.Parameter(torch.empty(1, num_student_tokens, proj_dim))
        nn.init.xavier_uniform_(self.queries)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, teacher_tokens: torch.Tensor) -> torch.Tensor:
        B = teacher_tokens.shape[0]

        kv = self.teacher_proj(teacher_tokens)
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        aligned_tokens = self.norm(queries + attn_out)

        return aligned_tokens


class BASDLoss(nn.Module):
    def __init__(
        self,
        base_criterion: nn.Module,
        student_dim: int,
        teacher_dim: int,
        student_depth: int,
        num_student_tokens: int,
        cross_attn_num_heads: int,
        *,
        config,
        student_num_heads: int,
        teacher_num_heads: int,
    ):
        super().__init__()
        self.base_criterion = base_criterion

        num_points = config.num_extraction_points
        if num_points == 1:
            self.token_layers = [student_depth - 1]
        else:
            self.token_layers = [round(i * (student_depth - 1) / (num_points - 1)) for i in range(num_points)]

        self.disabled_components = frozenset(config.disabled_components)

        self.cross_attn_projectors = nn.ModuleList([
            CrossAttentionProjector(
                num_student_tokens=num_student_tokens,
                teacher_dim=teacher_dim,
                proj_dim=teacher_dim,
                num_heads=cross_attn_num_heads,
            )
            for _ in self.token_layers
        ])

        self.rsd_loss = RedundancySuppressionLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            num_layers=len(self.token_layers),
            kappa=config.rsd_kappa,
        )
        self.attn_loss = AttentionDistillationLoss(
            student_heads_per_layer=student_num_heads,
            teacher_heads_per_layer=teacher_num_heads,
            num_layers=len(self.token_layers),
        )
        self.layer_selector = GrassmannianLayerSelector(
            num_extraction_points=len(self.token_layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            fixed_rank=config.layer_selector_fixed_rank,
        )

        self.uwso_temperature = config.uwso_temperature

    @torch.no_grad()
    def project_to_stiefel(self) -> None:
        for proj in self.cross_attn_projectors:
            _retract_to_stiefel(proj.teacher_proj)

    def _dual_space_geometric_loss(
        self,
        student_tokens: torch.Tensor,
        aligned_tokens: torch.Tensor,
        teacher_attn: torch.Tensor,
    ) -> torch.Tensor:
        """Unified sample-space (Procrustes) + feature-space (BW) loss.

        Auto-balances via loss ratio: alpha = L_feature / (L_sample + L_feature).
        When feature-space loss is large, alpha → 1, emphasizing sample-space alignment.
        When sample-space loss is large, alpha → 0, emphasizing feature-space alignment.
        """
        s = student_tokens.float()
        t = aligned_tokens.float()

        L_sample = geometric_relational_loss(s, t, teacher_attn)
        L_feature = bures_wasserstein_loss(s, t)

        with torch.no_grad():
            alpha = L_feature / (L_sample + L_feature + 1e-8)

        return alpha * L_sample + (1.0 - alpha) * L_feature

    def forward(
        self,
        student_output: torch.Tensor,
        targets: torch.Tensor,
        student_intermediates: dict[int, torch.Tensor],
        student_attns: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.base_criterion(student_output, targets)

        mixed_tokens, mixed_attns = self.layer_selector(
            student_intermediates, all_teacher_tokens, all_teacher_attns,
            self.token_layers,
        )

        aligned_tokens = {}
        for i, layer_idx in enumerate(self.token_layers):
            aligned_tokens[layer_idx] = self.cross_attn_projectors[i](
                mixed_tokens[layer_idx]
            )

        raw_losses = {}
        loss_info = {"ce_loss": ce_loss.item()}

        rsd_loss = self.rsd_loss(
            student_intermediates, aligned_tokens, self.token_layers
        )
        loss_info["rsd_loss"] = rsd_loss.item()
        if "rsd" not in self.disabled_components:
            raw_losses["rsd"] = rsd_loss

        dsgt_losses = [
            self._dual_space_geometric_loss(
                student_intermediates[layer_idx], aligned_tokens[layer_idx],
                mixed_attns[layer_idx],
            )
            for layer_idx in self.token_layers
        ]
        dsgt_loss = torch.stack(dsgt_losses).mean()
        loss_info["dsgt_loss"] = dsgt_loss.item()
        if "dsgt" not in self.disabled_components:
            raw_losses["dsgt"] = dsgt_loss

        attn_loss = self.attn_loss(student_attns, mixed_attns, self.token_layers)
        loss_info["attn_loss"] = attn_loss.item()
        if "attn" not in self.disabled_components:
            raw_losses["attn"] = attn_loss

        # Inverse-loss softmax weighting (UWSO): w_k = softmax(1/sg[L_k] / T)
        vals = list(raw_losses.values())
        inv = torch.stack([1.0 / v.detach() for v in vals])
        w = F.softmax(inv / self.uwso_temperature, dim=0)
        weighted_sum = sum(w[i] * vals[i] for i in range(len(vals)))

        total_loss = ce_loss + weighted_sum

        return total_loss, loss_info
