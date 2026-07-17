"""
models/cae.py

WaferCAE — простой сверточный автокодировщик, используемый в методе
CBAM-CNN для аугментации миноритарных классов (статья, раздел "Математика
автокодировщика"): единственная свёртка кодирует изображение в латентное
представление, зашумление латента (Z_noisy = Z + eps, eps ~ N(0, sigma^2))
и декодирование даёт синтетический, но реалистичный новый образец того же
класса. Не является частью тренировочного цикла метода — используется как
одноразовый шаг подготовки данных (см. datasets.prepare_datasets_cbam).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferCAE(nn.Module):
    """
    Z = sigma(We*X + be) ∈ R^{13x13x64} при input_size=26 (conv k=4,s=2,p=1: 26->13).
    X_hat = sigma(Wd*Z + bd) (ConvTranspose k=4,s=2,p=1: 13->26).
    """

    def __init__(self, input_channels: int = 3, latent_channels: int = 64, input_size: int = 26):
        super().__init__()
        self.output_size = input_size
        self.encoder_conv = nn.Conv2d(input_channels, latent_channels, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.ConvTranspose2d(latent_channels, input_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.encoder_conv(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = torch.sigmoid(self.decoder_conv(z))
        if recon.shape[-1] != self.output_size or recon.shape[-2] != self.output_size:
            recon = F.interpolate(recon, size=(self.output_size, self.output_size), mode="nearest")
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
