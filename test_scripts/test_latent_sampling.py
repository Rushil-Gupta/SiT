"""
End-to-end check that latent-space sampling (sample.py --use-latent) produces
a valid decoded image: runs sample.py as a subprocess and checks the output
file's shape/value range. Visual inspection of the actual cell morphology is
left to the user -- this only catches wiring bugs (wrong crop size, decode
producing NaNs/constant output, etc.), not sample quality.

Usage:
    python test_scripts/test_latent_sampling.py \\
        --ckpt /path/to/sit_checkpoint.pt \\
        --latent-dir /path/to/latent_dir \\
        --vae-checkpoint /path/to/finetuned_vae.pt \\
        --image-size 100
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image


def test_latent_sampling(ckpt, latent_dir, vae_checkpoint, image_size, output_path, num_sampling_steps):
    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        sys.executable, "sample.py", "ODE",
        "--ckpt", ckpt,
        "--use-latent",
        "--latent-dir", latent_dir,
        "--vae-checkpoint", vae_checkpoint,
        "--image-size", str(image_size),
        "--num-sampling-steps", str(num_sampling_steps),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"sample.py exited with code {result.returncode}")

    assert os.path.exists("sample.png"), "sample.py did not produce sample.png"

    img = np.array(Image.open("sample.png").convert("RGB")).astype(np.float32)
    print(f"  sample.png shape: {img.shape}, min={img.min():.1f}, max={img.max():.1f}, std={img.std():.2f}")

    assert img.std() > 1.0, \
        "Decoded sample grid has near-zero variance -- likely a blank/constant image (decode or crop bug)"
    assert not np.isnan(img).any(), "Decoded sample contains NaNs"

    print("  [PASS] sample.py --use-latent produced a non-degenerate image.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="SiT checkpoint trained with --use-latent")
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--num-sampling-steps", type=int, default=50,
                         help="Keep low for a quick wiring check, not sample quality")
    args = parser.parse_args()

    print("=== Latent-space sampling end-to-end test ===")
    test_latent_sampling(
        args.ckpt, args.latent_dir, args.vae_checkpoint, args.image_size,
        "sample.png", args.num_sampling_steps,
    )


if __name__ == "__main__":
    main()
