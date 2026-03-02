import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDistillationLoss(nn.Module):
    def __init__(
        self,
        student_heads_per_layer: int,
        teacher_heads_per_layer: int,
        num_layers: int,
        *,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        self.student_heads_per_layer = student_heads_per_layer
        self.teacher_heads_per_layer = teacher_heads_per_layer
        self.num_layers = num_layers

        total_student_heads = student_heads_per_layer * num_layers
        total_teacher_heads = teacher_heads_per_layer * num_layers

        self.head_aligner = nn.Conv2d(
            total_student_heads, total_teacher_heads,
            kernel_size=1, bias=False,
        )
        nn.init.kaiming_normal_(self.head_aligner.weight, mode="fan_out", nonlinearity="linear")
        self._raw_temperature = nn.Parameter(
            torch.tensor(math.log(math.exp(init_temperature) - 1.0))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self._raw_temperature)

    def _align_resolution(self, attn: torch.Tensor, target_size: int) -> torch.Tensor:
        B, H, N, _ = attn.shape
        if N == target_size:
            return attn
        attn_flat = attn.reshape(B * H, 1, N, N)
        attn_resized = F.interpolate(
            attn_flat, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        attn_resized = attn_resized.reshape(B, H, target_size, target_size)
        return attn_resized / attn_resized.sum(dim=-1, keepdim=True)

    def forward(
        self,
        student_attns: dict[int, torch.Tensor],
        teacher_attns: dict[int, torch.Tensor],
        layer_indices: list[int],
    ) -> torch.Tensor:
        s_stacked = torch.cat(
            [student_attns[layer] for layer in layer_indices], dim=1
        )

        aligned_logits = self.head_aligner(s_stacked)
        N_s = s_stacked.shape[2]

        aligned_per_layer = aligned_logits.split(self.teacher_heads_per_layer, dim=1)

        losses = []
        for i, layer in enumerate(layer_indices):
            t_attn = self._align_resolution(teacher_attns[layer], N_s)
            s_log_prob = F.log_softmax(aligned_per_layer[i] / self.temperature, dim=-1)
            losses.append(F.kl_div(s_log_prob, t_attn, reduction="batchmean"))

        return torch.stack(losses).mean()
