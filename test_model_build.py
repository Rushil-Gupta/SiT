"""
Dry-run test for SiT model in pixel-space diffusion mode.
Verifies model instantiation, forward pass, and output shapes
without any VAE/latent space dependencies.
"""
import torch
from models import SiT_models


def test_model_build():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test configuration: 4-channel, 100x100 pixel-space images
    image_size = 100
    in_channels = 4
    num_classes = 1000
    batch_size = 2

    # Test all model variants with patch sizes that divide 100 evenly
    variants_to_test = [
        ("SiT-S/2", 2),
        ("SiT-S/4", 4),
        ("SiT-S/5", None),  # patch_size=5 not in default registry, skip
        ("SiT-B/2", 2),
        ("SiT-B/4", 4),
        ("SiT-XL/2", 2),
    ]

    for model_name, patch_size in variants_to_test:
        if patch_size is None:
            continue

        print(f"\n--- Testing {model_name} (patch_size={patch_size}) ---")

        # Build model
        model = SiT_models[model_name](
            input_size=image_size,
            num_classes=num_classes,
            in_channels=in_channels,
            learn_sigma=False,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Create pixel-space input: (B, 4, 100, 100)
        x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
        t = torch.rand(batch_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        # Forward pass
        out = model(x, t, y)
        print(f"  Input shape:  {tuple(x.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
        assert out.shape == (batch_size, in_channels, image_size, image_size), \
            f"Expected output shape {(batch_size, in_channels, image_size, image_size)}, got {out.shape}"
        print(f"  Output matches expected shape!")

        # Test forward_with_cfg (classifier-free guidance)
        # Need to double batch for CFG: half conditional, half unconditional
        x_cfg = torch.randn(batch_size * 2, in_channels, image_size, image_size, device=device)
        t_cfg = torch.rand(batch_size * 2, device=device)
        y_cfg = torch.cat([y, torch.tensor([num_classes] * batch_size, device=device)])

        try:
            out_cfg = model.forward_with_cfg(x_cfg, t_cfg, y_cfg, cfg_scale=4.0)
            print(f"  CFG output shape: {tuple(out_cfg.shape)}")
            assert out_cfg.shape == (batch_size * 2, in_channels, image_size, image_size), \
                f"Expected CFG output shape {(batch_size * 2, in_channels, image_size, image_size)}, got {out_cfg.shape}"
            print(f"  CFG output matches expected shape!")
        except torch.OutOfMemoryError:
            print(f"  CFG test skipped (OOM - GPU memory constrained)")

        del model, x, x_cfg
        torch.cuda.empty_cache()

    print("\n=== All tests passed! Model build is correct for pixel-space diffusion. ===")


if __name__ == "__main__":
    test_model_build()
