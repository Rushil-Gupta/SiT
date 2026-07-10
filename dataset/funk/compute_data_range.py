"""
Compute per-channel min/max across all OPS images.
Reads raw .npy files, tracks running min/max per channel in float64 for precision.
Also counts images with constant channels (range=0) which cause NaN in current Arcsinh.
"""
import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing ops_dataset.npz")
    parser.add_argument("--report-every", type=int, default=1000,
                        help="Print progress every N images")
    args = parser.parse_args()

    npz_path = os.path.join(args.data_dir, 'ops_dataset.npz')
    data = np.load(npz_path, allow_pickle=True)
    file_paths = data['file_paths']
    print(f"Found {len(file_paths)} images")

    ch_min = np.full(4, float('inf'), dtype=np.float64)
    ch_max = np.full(4, float('-inf'), dtype=np.float64)
    const_occurrences = 0
    const_examples = []
    i = 1131604
    count = 45
    # for i, path in enumerate(file_paths):
    while i < len(file_paths) and count > 0:
        img = np.load(file_paths[i]).astype(np.float32)
        for c in range(4):
            lo = img[c].min()
            hi = img[c].max()
            ch_min[c] = min(ch_min[c], lo)
            ch_max[c] = max(ch_max[c], hi)
            if lo == hi:
                # const_occurrences += 1
                count -= 1
                # if len(const_examples) < 10:
                # const_examples.append((i, c, float(lo)))
                print(f"  Image {i}  channel {c}  constant={lo:.1f}")
        i += 1

        # if (i + 1) % args.report_every == 0:
        #     print(f"  [{i+1}/{len(file_paths)}]  per-ch min: {ch_min}  max: {ch_max}")

    # print(f"\n=== Final per-channel stats (raw uint16->float32) ===")
    # for c in range(4):
    #     print(f"  Channel {c}:  min={ch_min[c]:.1f}  max={ch_max[c]:.1f}  "
    #           f"range={ch_max[c]-ch_min[c]:.1f}")

    # print(f"\n=== Constant-channel images (range=0) ===")
    # print(f"  Total occurrences: {const_occurrences}")
    # for idx, c, val in const_examples:
    #     print(f"  Image {idx}  channel {c}  constant={val:.1f}")


if __name__ == "__main__":
    main()
