"""
FunkVAE: the pretrained Stable Diffusion VAE (AutoencoderKL), adapted to
4-channel, non-RGB Funk22 microscopy images.

The pretrained VAE (e.g. stabilityai/sd-vae-ft-mse) expects 3-channel RGB
input in [-1, 1] and produces a 4-channel latent at 1/8 spatial resolution.
Our data is 4-channel (not RGB) at 100x100. Two adaptations are needed:

  1. Spatial: 100 is not divisible by 8, and the resulting latent size must
     also divide evenly by SiT's patch_size (2, 4, or 8). Padding to 104
     gives a 13x13 latent, which is prime and fails all patch sizes, so we
     reflect-pad images to 128x128 before encoding (giving a clean 16x16
     latent, divisible by 2/4/8) and center-crop 128->100 after decoding.
  2. Channels: conv_in (encoder's first layer) and conv_out (decoder's last
     layer) are widened from 3 to 4 channels by copying the pretrained
     weights into the first 3 channels and initializing the 4th channel as
     the mean of the other 3. The rest of the pretrained weights are left
     untouched (and are expected to be finetuned afterwards via
     finetune_vae.py, since the pretrained encoder/decoder were trained on
     natural RGB photos and have no built-in familiarity with fluorescence
     microscopy channels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL


class FunkVAE(nn.Module):
    PRETRAINED_VAE = "stabilityai/sd-vae-ft-mse"
    NUM_CHANNELS = 4
    # 128 (not just the smallest multiple of 8 >= 100, which would be 104) so
    # that the resulting 16x16 latent divides evenly by every available SiT
    # patch_size (2, 4, 8) -- a 13x13 latent (from padding to 104) is prime
    # and fails all of them.
    PAD_TARGET = 128

    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(self.PRETRAINED_VAE)
        self._widen_conv_in()
        self._widen_conv_out()
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(state_dict["vae"] if "vae" in state_dict else state_dict)
        self.to(device)

    def _widen_conv_in(self):
        conv = self.vae.encoder.conv_in
        assert conv.in_channels == 3
        new_conv = nn.Conv2d(
            self.NUM_CHANNELS, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, bias=conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = conv.weight
            mean_weight = conv.weight.mean(dim=1, keepdim=True)
            for c in range(3, self.NUM_CHANNELS):
                new_conv.weight[:, c:c + 1] = mean_weight
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        self.vae.encoder.conv_in = new_conv
        self.vae.config.in_channels = self.NUM_CHANNELS

    def _widen_conv_out(self):
        conv = self.vae.decoder.conv_out
        assert conv.out_channels == 3
        new_conv = nn.Conv2d(
            conv.in_channels, self.NUM_CHANNELS, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, bias=conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:3] = conv.weight
            mean_weight = conv.weight.mean(dim=0, keepdim=True)
            for c in range(3, self.NUM_CHANNELS):
                new_conv.weight[c:c + 1] = mean_weight
            if conv.bias is not None:
                new_conv.bias[:3] = conv.bias
                mean_bias = conv.bias.mean()
                for c in range(3, self.NUM_CHANNELS):
                    new_conv.bias[c] = mean_bias
        self.vae.decoder.conv_out = new_conv
        self.vae.config.out_channels = self.NUM_CHANNELS

    @staticmethod
    def pad(x: torch.Tensor, target: int = PAD_TARGET) -> torch.Tensor:
        """Reflect-pad the last two (H, W) dims of x up to `target` on each side."""
        h, w = x.shape[-2], x.shape[-1]
        pad_h, pad_w = target - h, target - w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        return F.pad(x, (left, right, top, bottom), mode="reflect")

    @staticmethod
    def crop(x: torch.Tensor, size: int) -> torch.Tensor:
        """Center-crop the last two (H, W) dims of x down to `size`."""
        h, w = x.shape[-2], x.shape[-1]
        top, left = (h - size) // 2, (w - size) // 2
        return x[..., top:top + size, left:left + size]

    def encode(self, x: torch.Tensor, sample: bool = True):
        """x is assumed already padded to PAD_TARGET and normalized to [-1, 1].
        Returns (z, posterior); posterior.kl() gives the KL loss term."""
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() if sample else posterior.mode()
        return z, posterior

    def decode(self, z: torch.Tensor, crop_size: int = None) -> torch.Tensor:
        recon = self.vae.decode(z).sample
        if crop_size is not None:
            recon = self.crop(recon, crop_size)
        return recon
