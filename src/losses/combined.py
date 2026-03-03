import torch
import torch.nn as nn

from src.losses.attention import AttentionDistillationLoss
from src.losses.layer_selector import (
    GrassmannianLayerSelector,
    _retract_to_stiefel,
)
from src.losses.relational import geometric_relational_loss
from src.losses.rsd import RedundancySuppressionLoss


class CrossAttentionProjector(nn.Module):
    def __init__(self, num_student_tokens: int, teacher_dim: int, *, num_heads: int):
        super().__init__()
        self.teacher_proj = nn.Linear(teacher_dim, teacher_dim)
        self.queries = nn.Parameter(torch.empty(1, num_student_tokens, teacher_dim))
        nn.init.xavier_uniform_(self.queries)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=teacher_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(teacher_dim)
        self.output_proj = nn.Linear(teacher_dim, teacher_dim, bias=False)
        nn.init.orthogonal_(self.output_proj.weight)

    def forward(self, teacher_tokens: torch.Tensor) -> torch.Tensor:
        B = teacher_tokens.shape[0]

        kv = self.teacher_proj(teacher_tokens)
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        normed = self.norm(queries + attn_out)

        return self.output_proj(normed)


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
        student_heads_per_layer: list[int],
        teacher_heads_per_layer: list[int],
        teacher_has_cls_token: bool,
        teacher_feature_format: str,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_has_cls_token = teacher_has_cls_token

        if config.num_extraction_points == 1:
            self.token_layers = [student_depth - 1]
        else:
            self.token_layers = [
                round(i * (student_depth - 1) / (config.num_extraction_points - 1))
                for i in range(config.num_extraction_points)
            ]

        self.use_attn = teacher_feature_format == "token"

        self.cross_attn_projectors = nn.ModuleList([
            CrossAttentionProjector(
                num_student_tokens=num_student_tokens,
                teacher_dim=teacher_dim,
                num_heads=cross_attn_num_heads,
            )
            for _ in self.token_layers
        ])

        self.rsd_loss = RedundancySuppressionLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            num_layers=len(self.token_layers),
            kappa=1.0 / teacher_dim,
        )

        if self.use_attn:
            self.attn_loss = AttentionDistillationLoss(
                student_heads_per_layer=[
                    student_heads_per_layer[l] for l in self.token_layers
                ],
                teacher_heads_per_layer=[
                    teacher_heads_per_layer[l] for l in self.token_layers
                ],
                num_layers=len(self.token_layers),
            )

        self.layer_selector = GrassmannianLayerSelector(
            num_extraction_points=len(self.token_layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        )

    @torch.no_grad()
    def project_to_stiefel(self) -> None:
        for proj in self.cross_attn_projectors:
            _retract_to_stiefel(proj.output_proj)

    def forward(
        self,
        student_output: torch.Tensor,
        targets: torch.Tensor,
        student_intermediates: dict[int, torch.Tensor],
        student_attns: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
    ) -> torch.Tensor:
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

        rsd_loss = self.rsd_loss(
            student_intermediates, aligned_tokens, self.token_layers
        )

        geo_losses = []
        for layer_idx in self.token_layers:
            geo_losses.append(geometric_relational_loss(
                student_intermediates[layer_idx], aligned_tokens[layer_idx],
                mixed_attns[layer_idx],
                has_cls_token=self.teacher_has_cls_token,
            ))
        geo_loss = torch.stack(geo_losses).mean()

        vals = [ce_loss, rsd_loss, geo_loss]

        if self.use_attn:
            vals.append(self.attn_loss(student_attns, mixed_attns, self.token_layers))

        # UW-SO weighting (Kirchdorfer et al. 2024): w_i = (1/L_i) / Σ(1/L_j)
        eps = torch.finfo(vals[0].dtype).eps
        inv = torch.stack([1.0 / v.detach().clamp(min=eps) for v in vals])
        w = inv / inv.sum()

        return sum(w[i] * vals[i] for i in range(len(vals)))
