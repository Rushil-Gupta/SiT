import torch


def rescale_intensity(arr, bounds=(0.5, 99.5), out_range=(0.0, 1.0)):
    arr = arr.float() / 255
    flat = arr.flatten()[::100]
    percentiles = torch.quantile(
        flat, torch.tensor([bounds[0] / 100.0, bounds[1] / 100.0], device=arr.device)
    )
    arr = torch.clamp(arr, percentiles[0], percentiles[1])
    rng = percentiles[1] - percentiles[0]
    arr = (arr - percentiles[0]) / rng.clamp(min=1e-8)
    arr = arr * (out_range[1] - out_range[0]) + out_range[0]
    return arr


def to_rgb(img, dtype=torch.float32):
    num_channels_required = 6
    b, num_channels, length, width = img.shape
    prepped_img = torch.zeros(
        b, num_channels_required, length, width, dtype=img.dtype, device=img.device
    )
    if num_channels < num_channels_required:
        prepped_img[:, :num_channels, :, :] += img
    elif num_channels > num_channels_required:
        prepped_img += img[:, :num_channels_required, :, :]
    else:
        prepped_img += img

    blue = torch.tensor([1, 0, 0], dtype=dtype, device=img.device)
    green = torch.tensor([0, 1, 0], dtype=dtype, device=img.device)
    red = torch.tensor([0, 0, 1], dtype=dtype, device=img.device)
    cyan = torch.tensor([1, 1, 0], dtype=dtype, device=img.device)
    magenta = torch.tensor([1, 0, 1], dtype=dtype, device=img.device)
    yellow = torch.tensor([0, 1, 1], dtype=dtype, device=img.device)
    rgb_map = torch.stack([blue, green, red, cyan, magenta, yellow])

    rgb_img = torch.einsum("nchw,ct->nthw", prepped_img.to(dtype=dtype), rgb_map) / 3.0
    return rescale_intensity(rgb_img, bounds=(0.1, 99.9))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


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
