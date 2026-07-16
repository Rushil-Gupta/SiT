import os
import json
import torch
import numpy as np
from PIL import Image


@torch.no_grad()
def _decode_latents(samples, vae, scale_factor, pixel_size):
    """Decode raw VAE latents (model output) back to a [0,1] pixel-space image batch.

    Mirrors the pattern in sample.py/train.py: vae.decode() expects latents
    divided by the dataset's scale_factor, and its output is nominally in
    [-1, 1] (the VAE was trained on to_pm1-normalized pixel data), so it must
    be rescaled to [0,1] to match what save_samples/extractors expect (the
    same range as pixel-space model output and GlobalMinMaxNorm-normalized
    real images).
    """
    # `samples` is already on the sampling device, which is where `vae` lives too
    # (both constructed from the same `device` arg in the caller) — no .to() needed.
    decoded = vae.decode(samples / scale_factor, crop_size=pixel_size)
    return ((decoded + 1) / 2).clamp(0, 1)


@torch.no_grad()
def generate_balanced_samples(model_fn, transport_sampler, num_classes, null_idx,
                              n_per_class, cfg_scale, device, image_size=100,
                              ode_method="dopri5", ode_steps=50,
                              vae=None, scale_factor=None, pixel_image_size=None):
    """
    image_size: spatial size of the noise tensor fed to the model (the latent
        size, e.g. 16, when `vae` is given; otherwise the pixel image size).
    vae, scale_factor, pixel_image_size: if `vae` is provided, samples are
        decoded back to a `pixel_image_size` x `pixel_image_size` pixel-space
        [0,1] image via vae.decode(sample / scale_factor) before being
        returned — required for latent-space (--use-latent) models, since
        extractors/FID/save_samples all expect real pixel-space images.
    """
    if vae is not None:
        assert scale_factor is not None and pixel_image_size is not None, \
            "scale_factor and pixel_image_size are required when vae is provided"

    all_images = []
    all_labels = []
    sample_fn = transport_sampler.sample_ode(
        sampling_method=ode_method, num_steps=ode_steps
    )

    for c in range(num_classes):
        ys = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        z = torch.randn(n_per_class, 4, image_size, image_size, device=device)
        using_cfg = cfg_scale > 1.0
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.full((n_per_class,), null_idx, dtype=torch.long, device=device)
            ys = torch.cat([ys, y_null], 0)
            model_kwargs = dict(y=ys, cfg_scale=cfg_scale)
        else:
            model_kwargs = dict(y=ys)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        if vae is not None:
            samples = _decode_latents(samples, vae, scale_factor, pixel_image_size)
        all_images.append(samples.cpu())
        all_labels.append(torch.full((n_per_class,), c, dtype=torch.long))

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels


@torch.no_grad()
def generate_class_samples(model_fn, transport_sampler, class_id, n_samples,
                           null_idx, cfg_scale, device, image_size=100,
                           ode_method="dopri5", ode_steps=50,
                           vae=None, scale_factor=None, pixel_image_size=None):
    if vae is not None:
        assert scale_factor is not None and pixel_image_size is not None, \
            "scale_factor and pixel_image_size are required when vae is provided"

    sample_fn = transport_sampler.sample_ode(
        sampling_method=ode_method, num_steps=ode_steps
    )
    ys = torch.full((n_samples,), class_id, dtype=torch.long, device=device)
    z = torch.randn(n_samples, 4, image_size, image_size, device=device)
    using_cfg = cfg_scale > 1.0
    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.full((n_samples,), null_idx, dtype=torch.long, device=device)
        ys = torch.cat([ys, y_null], 0)
        model_kwargs = dict(y=ys, cfg_scale=cfg_scale)
    else:
        model_kwargs = dict(y=ys)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)
    if vae is not None:
        samples = _decode_latents(samples, vae, scale_factor, pixel_image_size)
    return samples.cpu()


def save_samples(images, labels, save_dir, perturbation_map=None,
                 selected_original_indices=None):
    os.makedirs(save_dir, exist_ok=True)
    unique_labels = labels.unique()
    for c in unique_labels:
        class_dir = os.path.join(save_dir, f"class_{int(c):05d}")
        os.makedirs(class_dir, exist_ok=True)

    if perturbation_map is not None:
        with open(os.path.join(save_dir, "perturbation_map.json"), "w") as f:
            json.dump({str(k): v for k, v in perturbation_map.items()}, f, indent=2)

    # Use the same pseudo-coloring the extractors see (ConvtoRGB), rather than
    # just dropping channel 3, so saved PNGs visually match what FID/PR/MMD/KID
    # were actually computed on.
    if images.shape[1] == 4:
        from .utils import ConvtoRGB
        rgb_images = ConvtoRGB(in_channels=4, out_channels=3)(images)
    else:
        rgb_images = images

    for i in range(len(images)):
        img = rgb_images[i]
        label = int(labels[i])
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        save_path = os.path.join(save_dir, f"class_{label:05d}", f"{i:06d}.png")
        Image.fromarray(img_np).save(save_path)

    config = {"num_images": len(images), "num_classes": len(unique_labels)}
    if selected_original_indices is not None:
        config["selected_original_indices"] = list(selected_original_indices)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_samples(save_dir):
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    images = []
    labels = []
    for c_dir in sorted(os.listdir(save_dir)):
        if not c_dir.startswith("class_"):
            continue
        label = int(c_dir.split("_")[1])
        c_path = os.path.join(save_dir, c_dir)
        for fname in sorted(os.listdir(c_path)):
            if not fname.endswith(".png"):
                continue
            img = Image.open(os.path.join(c_path, fname))
            img_np = np.array(img).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)
            images.append(img_t)
            labels.append(label)

    perm_map = None
    perm_path = os.path.join(save_dir, "perturbation_map.json")
    if os.path.exists(perm_path):
        with open(perm_path) as f:
            perm_map = {int(k): v for k, v in json.load(f).items()}

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    selected_original_indices = config.get("selected_original_indices", None)
    return images, labels, perm_map, selected_original_indices
