import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class ConvtoRGB(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def rescale_intensity(self, arr, bounds=(0.5, 99.5), out_range=(0.0, 1.0)):
        flat = arr.flatten()[::100]
        percentiles = torch.quantile(
            flat, torch.tensor([bounds[0] / 100.0, bounds[1] / 100.0], device=arr.device)
        )
        arr = torch.clamp(arr, percentiles[0], percentiles[1])
        rng = percentiles[1] - percentiles[0]
        arr = (arr - percentiles[0]) / rng.clamp(min=1e-8)
        arr = arr * (out_range[1] - out_range[0]) + out_range[0]
        return arr

    def to_rgb(self, img, dtype=torch.float32):
        c1 = torch.tensor([1, 1, 0], dtype=dtype, device=img.device)
        c2 = torch.tensor([1, 0, 1], dtype=dtype, device=img.device)
        c3 = torch.tensor([0, 1, 1], dtype=dtype, device=img.device)
        c4 = torch.tensor([1, 1, 1], dtype=dtype, device=img.device)
        rgb_map = torch.stack([c1, c2, c3, c4])

        rgb_img = torch.einsum("nchw,ct->nthw", img.to(dtype=dtype), rgb_map) / 3.0
        return self.rescale_intensity(rgb_img, bounds=(0.1, 99.9))

    def forward(self, x):
        return self.to_rgb(x)
    
class ImageClamper(torch.nn.Module):
    _min = 0.0
    _max = 1.0
    def __init__(self, minv=0.0, maxv=1.0):
        super().__init__()
        self._min = minv
        self._max = maxv

    def forward(self, x):
        return torch.clamp(x, self._min, self._max)

class MinMaxNormalize(torch.nn.Module):
    _ch_max = torch.Tensor([65535.0, 16628.0, 65212.0, 65535.0])
    _ch_min = torch.Tensor([0.0, 0.0, 0.0, 0.0])

    def forward(self, x):
        ch_range = self._ch_max[:, None, None] - self._ch_min[:, None, None]
        return (x - self._ch_min[:, None, None]) / ch_range.to(x.device)


def invert_minmax(x):
    ch_range = (MinMaxNormalize._ch_max - MinMaxNormalize._ch_min).to(x.device)
    ch_min = MinMaxNormalize._ch_min.to(x.device)
    return x * ch_range[None, :, None, None] + ch_min[None, :, None, None]
