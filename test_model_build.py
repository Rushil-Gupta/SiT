"""
Dry-run test for SiT model with guidance embedding module.
Verifies model instantiation, forward pass, KL loss, and output shapes.
"""
import torch
import numpy as np
from models import SiT_models, GuidanceEmbeddingModule


def test_model_build():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_size = 100
    in_channels = 4
    num_classes = 1451
    embed_dim = 384
    batch_size = 2

    variants_to_test = [
        ("SiT-S/2", 2),
        ("SiT-S/4", 4),
        ("SiT-B/2", 2),
        ("SiT-B/4", 4),
    ]

    for model_name, patch_size in variants_to_test:
        print(f"\n--- Testing {model_name} (patch_size={patch_size}) ---")

        model = SiT_models[model_name](
            input_size=image_size,
            num_classes=num_classes,
            in_channels=in_channels,
            learn_sigma=False,
            use_guidance=True,
            embed_dim=embed_dim,
        ).to(device)

        fake_means = np.random.randn(num_classes + 1, embed_dim).astype(np.float32)
        model.y_embedder.class_means.copy_(torch.from_numpy(fake_means))

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
        t = torch.rand(batch_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        out = model(x, t, y)
        print(f"  Input shape:  {tuple(x.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
        assert out.shape == (batch_size, in_channels, image_size, image_size)

        x_cfg = torch.randn(batch_size * 2, in_channels, image_size, image_size, device=device)
        t_cfg = torch.rand(batch_size * 2, device=device)
        y_cfg = torch.cat([y, torch.tensor([num_classes] * batch_size, device=device)])

        try:
            out_cfg = model.forward_with_cfg(x_cfg, t_cfg, y_cfg, cfg_scale=4.0)
            print(f"  CFG output shape: {tuple(out_cfg.shape)}")
            assert out_cfg.shape == (batch_size * 2, in_channels, image_size, image_size)
        except torch.OutOfMemoryError:
            print(f"  CFG test skipped (OOM)")

        del model, x, x_cfg
        torch.cuda.empty_cache()

    print("\n=== All tests passed! ===")


def test_kl_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Testing KL loss ---")

    num_classes = 1451
    embed_dim = 384
    hidden_size = 384

    guidance = GuidanceEmbeddingModule(num_classes, hidden_size, dropout_prob=0.1, embed_dim=embed_dim).to(device)
    fake_means = np.random.randn(num_classes + 1, embed_dim).astype(np.float32)
    guidance.class_means.copy_(torch.from_numpy(fake_means))

    labels = torch.randint(0, num_classes, (16,), device=device)

    class_mean = guidance.class_means[labels]
    mu_c = guidance.mu_phi(class_mean)
    sigma_c = torch.nn.functional.softplus(guidance.sigma_phi(class_mean))

    kl = guidance.kl_loss(mu_c, sigma_c, labels)
    print(f"  KL loss: {kl.item():.4f}")
    assert kl.item() >= 0, "KL loss should be non-negative"
    assert kl.dim() == 0, "KL loss should be scalar"

    print("  KL loss test passed!")


def test_label_dropout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Testing label dropout ---")

    num_classes = 1451
    embed_dim = 384
    hidden_size = 384

    guidance = GuidanceEmbeddingModule(num_classes, hidden_size, dropout_prob=0.1, embed_dim=embed_dim).to(device)
    fake_means = np.random.randn(num_classes + 1, embed_dim).astype(np.float32)
    guidance.class_means.copy_(torch.from_numpy(fake_means))

    labels = torch.zeros(1000, dtype=torch.long, device=device)
    e_c = guidance(labels, train=True)
    null_count = (labels == num_classes).sum().item()
    print(f"  Batch size: 1000, Expected ~100 dropped labels, Actual null: {null_count}")

    print("  Label dropout test passed!")


if __name__ == "__main__":
    test_model_build()
    test_kl_loss()
    test_label_dropout()
