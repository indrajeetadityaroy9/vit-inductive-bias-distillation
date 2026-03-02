import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDistillationLoss(nn.Module):
    def __init__(
        self,
        student_heads_per_layer: list[int],
        teacher_heads_per_layer: list[int],
        num_layers: int,
        *,
        init_temperature: float = 1.0,
    ):
        super().__init__()

        self.head_aligners = nn.ModuleList([
            nn.Conv2d(
                student_heads_per_layer[i], teacher_heads_per_layer[i],
                kernel_size=1, bias=False,
            )
            for i in range(num_layers)
        ])
        for aligner in self.head_aligners:
            nn.init.kaiming_normal_(aligner.weight, mode="fan_out", nonlinearity="linear")

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
        losses = []
        for i, layer in enumerate(layer_indices):
            s_attn = student_attns[layer]
            t_attn = teacher_attns[layer]

            N_s = s_attn.shape[2]
            t_attn = self._align_resolution(t_attn, N_s)

            aligned = self.head_aligners[i](s_attn)
            s_log_prob = F.log_softmax(aligned / self.temperature, dim=-1)
            losses.append(F.kl_div(s_log_prob, t_attn, reduction="batchmean"))

        return torch.stack(losses).mean()
