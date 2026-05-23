"""
Test that KL loss doesn't explode during training with the fixed implementation.
Simulates many training steps and tracks KL loss behavior.
"""
import torch
import numpy as np
from models import GuidanceEmbeddingModule


def test_kl_stability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing KL loss stability on {device}")

    num_classes = 1451
    embed_dim = 384
    hidden_size = 384
    batch_size = 64

    guidance = GuidanceEmbeddingModule(num_classes, hidden_size, dropout_prob=0.1, embed_dim=embed_dim).to(device)
    fake_means = np.random.randn(num_classes + 1, embed_dim).astype(np.float32)
    guidance.class_means.copy_(torch.from_numpy(fake_means))

    optimizer = torch.optim.AdamW(
        list(guidance.mu_phi.parameters()) +
        list(guidance.sigma_phi.parameters()) +
        [guidance.mu_eta, guidance.Sigma_eta],
        lr=1e-4,
    )

    print(f"\nInitial Sigma_eta: min={guidance.Sigma_eta.min().item():.4f}, max={guidance.Sigma_eta.max().item():.4f}, mean={guidance.Sigma_eta.mean().item():.4f}")

    kl_history = []
    sigma_eta_history = []
    num_steps = 10000

    print(f"Running {num_steps} training steps...")
    for step in range(num_steps):
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        class_mean = guidance.class_means[labels]

        mu_c = guidance.mu_phi(class_mean)
        sigma_c = torch.nn.functional.softplus(guidance.sigma_phi(class_mean))

        null_mask = (labels == num_classes)
        sigma_c = sigma_c * (~null_mask).unsqueeze(-1).float()

        kl = guidance.kl_loss(mu_c, sigma_c, labels)

        if torch.isnan(kl) or torch.isinf(kl):
            print(f"\n*** NaN/Inf at step {step}! KL={kl.item()} ***")
            break

        kl_history.append(kl.item())
        sigma_eta_history.append(guidance.Sigma_eta.mean().item())

        optimizer.zero_grad()
        kl.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1:5d}: KL={kl.item():.4f}, Sigma_eta mean={guidance.Sigma_eta.mean().item():.6f}")

    print(f"\n--- Results ---")
    print(f"Steps completed: {len(kl_history)}")
    print(f"KL range: [{min(kl_history):.4f}, {max(kl_history):.4f}]")
    print(f"Final Sigma_eta: min={guidance.Sigma_eta.min().item():.6f}, max={guidance.Sigma_eta.max().item():.6f}, mean={guidance.Sigma_eta.mean().item():.6f}")
    print(f"Sigma_eta² range: [{(guidance.Sigma_eta**2).min().item():.6f}, {(guidance.Sigma_eta**2).max().item():.6f}]")

    if len(kl_history) == num_steps:
        print("\n*** SUCCESS: KL loss remained stable for all steps ***")
    else:
        print(f"\n*** FAILURE: KL exploded at step {len(kl_history)} ***")


def test_kl_values():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Testing KL loss values ---")

    num_classes = 1451
    embed_dim = 384
    hidden_size = 384

    guidance = GuidanceEmbeddingModule(num_classes, hidden_size, dropout_prob=0.0, embed_dim=embed_dim).to(device)
    fake_means = np.random.randn(num_classes + 1, embed_dim).astype(np.float32) * 0.1
    guidance.class_means.copy_(torch.from_numpy(fake_means))

    labels = torch.randint(0, num_classes, (32,), device=device)
    class_mean = guidance.class_means[labels]
    mu_c = guidance.mu_phi(class_mean)
    sigma_c = torch.nn.functional.softplus(guidance.sigma_phi(class_mean))

    kl = guidance.kl_loss(mu_c, sigma_c, labels)
    print(f"KL loss: {kl.item():.4f}")
    assert not torch.isnan(kl), "KL should not be NaN"
    assert not torch.isinf(kl), "KL should not be Inf"
    assert kl.item() >= 0, f"KL should be non-negative, got {kl.item()}"
    print("KL value test passed!")


if __name__ == "__main__":
    test_kl_values()
    test_kl_stability()
