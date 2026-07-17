"""
models/vae.py

WaferVAE — Variational Autoencoder для 64x64x1 wafer maps, используемый в
методе Wei et al. (2024) "Latent Vector Representation" для извлечения
глобального латентного вектора (fault distribution feature), который затем
вплавляется в ResNet backbone student-модели (см. FusionResNetBackbone).
"""

from typing import Tuple, Sequence

import torch
import torch.nn as nn


class WaferVAE(nn.Module):
    """
    Энкодер: N conv-блоков (stride 2), по умолчанию 64->32->16->8, затем
    FC -> (mu, logvar). Декодер: зеркальный transpose-conv путь 8->16->32->64,
    финальный sigmoid (wafer map нормализован в [0, 1]).
    """

    def __init__(self, latent_dim: int = 128, hidden_channels: Sequence[int] = (32, 64, 128), input_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.final_channels = hidden_channels[-1]
        self.feat_size = input_size // (2 ** len(hidden_channels))

        encoder_layers = []
        in_ch = 1
        for ch in hidden_channels:
            encoder_layers += [
                nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = ch
        self.encoder_conv = nn.Sequential(*encoder_layers)

        flat_dim = self.final_channels * self.feat_size * self.feat_size
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        decoder_channels = list(hidden_channels[::-1]) + [1]
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            in_ch, out_ch = decoder_channels[i], decoder_channels[i + 1]
            is_last = i == len(decoder_channels) - 2
            decoder_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if not is_last:
                decoder_layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.decoder_conv = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.final_channels, self.feat_size, self.feat_size)
        recon = self.decoder_conv(h)
        return torch.sigmoid(recon)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Детерминированное извлечение латентного вектора для fusion (mu, без сэмплирования шума)."""
        mu, _ = self.encode(x)
        return mu
