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
                student_heads_per_layer=student_heads_per_layer,
                teacher_heads_per_layer=teacher_heads_per_layer,
                num_layers=len(self.token_layers),
            )

        self.bw_student_proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.orthogonal_(self.bw_student_proj.weight)

        self.layer_selector = GrassmannianLayerSelector(
            num_extraction_points=len(self.token_layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        )

        self._raw_uwso_temperature = nn.Parameter(torch.zeros(1))

    @property
    def uwso_temperature(self) -> torch.Tensor:
        return F.softplus(self._raw_uwso_temperature)

    @torch.no_grad()
    def project_to_stiefel(self) -> None:
        for proj in self.cross_attn_projectors:
            _retract_to_stiefel(proj.output_proj)
        _retract_to_stiefel(self.bw_student_proj)

    def _dual_space_geometric_loss(
        self,
        student_tokens: torch.Tensor,
        aligned_tokens: torch.Tensor,
        teacher_attn: torch.Tensor,
    ) -> torch.Tensor:
        L_sample = geometric_relational_loss(
            student_tokens, aligned_tokens, teacher_attn,
            has_cls_token=self.teacher_has_cls_token,
        )
        L_feature = bures_wasserstein_loss(
            self.bw_student_proj(student_tokens), aligned_tokens,
        )

        return L_sample + L_feature

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

        loss_info = {"ce_loss": ce_loss.item()}

        rsd_loss = self.rsd_loss(
            student_intermediates, aligned_tokens, self.token_layers
        )
        loss_info["rsd_loss"] = rsd_loss.item()

        dsgt_losses = [
            self._dual_space_geometric_loss(
                student_intermediates[layer_idx], aligned_tokens[layer_idx],
                mixed_attns[layer_idx],
            )
            for layer_idx in self.token_layers
        ]
        dsgt_loss = torch.stack(dsgt_losses).mean()
        loss_info["dsgt_loss"] = dsgt_loss.item()

        vals = [rsd_loss, dsgt_loss]

        if self.use_attn:
            attn_loss = self.attn_loss(student_attns, mixed_attns, self.token_layers)
            loss_info["attn_loss"] = attn_loss.item()
            vals.append(attn_loss)
        else:
            loss_info["attn_loss"] = 0.0

        # Inverse-loss softmax weighting (UWSO): w_k = softmax(1/sg[L_k] / T)
        inv = torch.stack([1.0 / (v.detach() + 1e-8) for v in vals])
        w = F.softmax(inv / self.uwso_temperature, dim=0)
        K = len(vals)
        w = w.clamp(min=1.0 / (2 * K), max=2.0 / K)
        w = w / w.sum()
        weighted_sum = sum(w[i] * vals[i] for i in range(len(vals)))

        total_loss = ce_loss + weighted_sum

        return total_loss, loss_info
