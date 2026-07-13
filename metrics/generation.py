import os
import json
import torch
import numpy as np
from PIL import Image


@torch.no_grad()
def generate_balanced_samples(model_fn, transport_sampler, num_classes, null_idx,
                              n_per_class, cfg_scale, device, image_size=100,
                              ode_method="dopri5", ode_steps=50):
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
        all_images.append(samples.cpu())
        all_labels.append(torch.full((n_per_class,), c, dtype=torch.long))

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels


@torch.no_grad()
def generate_class_samples(model_fn, transport_sampler, class_id, n_samples,
                           null_idx, cfg_scale, device, image_size=100,
                           ode_method="dopri5", ode_steps=50):
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

    for i in range(len(images)):
        img = images[i]
        label = int(labels[i])
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
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
