"""
models/mm_wae.py

MM-WAE: Multimodal Wasserstein Autoencoder для wafer map defect recognition
(Zhang, Sun, Liu, Zhang, 2026, "MM-WAE: Multimodal Wasserstein Autoencoders
for Semi-Supervised Wafer Map Defect Recognition"). Три параллельные ветви
признаков (spatial, frequency-aware, texture) -> multi-head attention +
gated fusion -> детерминированный WAE-энкодер (без reparameterization,
регуляризация латента через MMD, а не KL как у VAE) -> decoder
(реконструкция) + мультимодальный классификатор поверх латентного z.

Интерпретационные допущения (статья не специфицирует однозначно):
  - Gated fusion применяется к POST-MHA признакам F' (после self-attention +
    residual), а не к исходным спроецированным f_tilde^(m) до attention —
    статья использует обозначение f_tilde и для входа гейтинга, и для
    финальной суммы, но архитектурно логичнее, чтобы гейтинг учитывал уже
    провзаимодействовавшие через attention представления.
  - Классификаторные ветви g^(m)(z) проецируют z в R^128 каждая;
    "конкатенация" из статьи реализована как ВЗВЕШЕННАЯ СУММА (не буквальная
    конкатенация, которая дала бы 384-мерный вектор, а не заявленные статьёй 128).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBranch(nn.Module):
    """
    3-стадийный conv-даунсэмплинг (4x4, stride2, pad1), 32/64/128 каналов.
    32x32 -> 4x4x128 -> Flatten -> FC -> f(s) в R^512.
    """

    def __init__(self, input_size: int = 32, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        final_size = input_size // 8
        self.fc = nn.Linear(128 * final_size * final_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)


class FrequencyBranch(nn.Module):
    """
    3x3 conv filter bank -> 4 feature maps {Fk}. GAP каждой -> 4-мерный вектор
    -> 2-layer FC -> softmax attention alpha в R^4. F_tilde_k = alpha_k * F_k.
    AdaptiveAvgPool(4x4) + flatten -> f(f) в R^64 (4 канала x 4x4 = 64).
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.filter_bank = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.attn_fc = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 4)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter_bank(x)  # [B, 4, H, W]
        gap = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # [B, 4]
        alpha = F.softmax(self.attn_fc(gap), dim=1)  # [B, 4]
        feats = feats * alpha.unsqueeze(-1).unsqueeze(-1)
        pooled = self.pool(feats)  # [B, 4, 4, 4]
        return torch.flatten(pooled, 1)  # [B, 64]


class TextureBranch(nn.Module):
    """
    3 conv-слоя (3x3, каналы 16/32/32) + attention-модуль (spatial gate, не
    специфицирован статьёй в деталях). AdaptiveAvgPool(2x2) + flatten ->
    f(t) в R^128 (32 канала x 2x2 = 128).
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.attn_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        gate = torch.sigmoid(self.attn_conv(feat))
        feat = feat * gate
        pooled = self.pool(feat)
        return torch.flatten(pooled, 1)  # [B, 128]


class MultimodalFusion(nn.Module):
    """
    Проекция каждой ветви в общее пространство d_model=256, multi-head
    self-attention между тремя модальностями (+ residual: F' = MHA(F) + F),
    затем gated fusion (softmax-веса важности по конкатенации F') -> итоговый
    fused вектор в R^d_model.

    Возвращает (fused, gate_weights) — gate_weights нужны для modality
    consistency loss (сравнение с branch weights классификатора).
    """

    def __init__(self, spatial_dim=512, freq_dim=64, texture_dim=128, d_model=256, num_heads=4):
        super().__init__()
        self.proj_s = nn.Linear(spatial_dim, d_model)
        self.proj_f = nn.Linear(freq_dim, d_model)
        self.proj_t = nn.Linear(texture_dim, d_model)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.gate_fc = nn.Linear(3 * d_model, 3)

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor, f_t: torch.Tensor):
        s = self.proj_s(f_s)
        fr = self.proj_f(f_f)
        t = self.proj_t(f_t)

        seq = torch.stack([s, fr, t], dim=1)  # [B, 3, d_model]
        attn_out, _ = self.mha(seq, seq, seq)
        fused_seq = attn_out + seq  # residual, формула F' = MHA(F) + F

        gate_input = fused_seq.flatten(1)  # [B, 3*d_model]
        gate_weights = F.softmax(self.gate_fc(gate_input), dim=1)  # [B, 3]

        fused = (fused_seq * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, d_model]
        return fused, gate_weights


class MMWAEDecoder(nn.Module):
    """z(64) -> FC -> h0 в R^{128x4x4} -> 3 transposed conv (64/32/1 каналов) -> Tanh."""

    def __init__(self, latent_dim: int = 64, output_size: int = 32):
        super().__init__()
        self.init_size = output_size // 8
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h0 = F.relu(self.fc(z))
        h0 = h0.view(-1, 128, self.init_size, self.init_size)
        recon = self.deconv(h0)
        return torch.tanh(recon)


class MultimodalClassifier(nn.Module):
    """
    Три ветви g^(s)/g^(f)/g^(t)(z) (2-layer FC, каждая проецирует z в R^128),
    взвешиваются глобальным learnable branch-weight вектором a (softmax),
    суммируются (см. докстринг модуля — "конкатенация" из статьи
    интерпретирована как взвешенная сумма) -> classification head -> logits.

    Возвращает (logits, branch_weights) — branch_weights нужны для modality
    consistency loss.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, num_classes: int = 9):
        super().__init__()

        def make_branch():
            return nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)
            )

        self.branch_s = make_branch()
        self.branch_f = make_branch()
        self.branch_t = make_branch()
        self.branch_weights_raw = nn.Parameter(torch.zeros(3))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor):
        g_s = self.branch_s(z)
        g_f = self.branch_f(z)
        g_t = self.branch_t(z)

        a = F.softmax(self.branch_weights_raw, dim=0)  # [3]
        h_c = a[0] * g_s + a[1] * g_f + a[2] * g_t
        logits = self.head(h_c)
        return logits, a.unsqueeze(0).expand(z.shape[0], -1)  # broadcast по батчу для сравнения с gate_weights


class MMWAE(nn.Module):
    """
    Полная модель: 3 ветви признаков -> fusion -> WAE encoder (детерминированный,
    регуляризация латента через MMD, не через reparameterization+KL как у VAE)
    -> decoder (реконструкция) + классификатор поверх z.

    forward(x) -> (logits, recon, z, gate_weights, branch_weights)
    """

    def __init__(
        self,
        num_classes: int = 9,
        input_size: int = 32,
        latent_dim: int = 64,
        spatial_dim: int = 512,
        freq_dim: int = 64,
        texture_dim: int = 128,
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        classifier_hidden_dim: int = 128,
    ):
        super().__init__()
        self.spatial_branch = SpatialBranch(input_size=input_size, out_dim=spatial_dim)
        self.frequency_branch = FrequencyBranch()
        self.texture_branch = TextureBranch()
        self.fusion = MultimodalFusion(
            spatial_dim=spatial_dim, freq_dim=freq_dim, texture_dim=texture_dim,
            d_model=fusion_dim, num_heads=fusion_heads,
        )
        self.encoder_fc = nn.Linear(fusion_dim, latent_dim)
        self.decoder = MMWAEDecoder(latent_dim=latent_dim, output_size=input_size)
        self.classifier = MultimodalClassifier(
            latent_dim=latent_dim, hidden_dim=classifier_hidden_dim, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor):
        f_s = self.spatial_branch(x)
        f_f = self.frequency_branch(x)
        f_t = self.texture_branch(x)

        fused, gate_weights = self.fusion(f_s, f_f, f_t)
        z = self.encoder_fc(fused)

        recon = self.decoder(z)
        logits, branch_weights = self.classifier(z)

        return logits, recon, z, gate_weights, branch_weights
