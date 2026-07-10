"""
Process OPS image data from funk22-npy.txt into a ready-to-use dataset.

Usage:
    python ops_data.py --data-dir /path/to/dir --build-db    # Create SQLite DB
    python ops_data.py --data-dir /path/to/dir               # Build dataset artifacts from DB

This script:
1. (Optional) Builds ops_files.db from funk22-npy.txt
2. Counts observations per perturbation
3. Filters to [4000, 5000] range, excludes nontargeting
4. Computes per-channel mean/std from all nontargeting images
5. Saves ops_dataset.npz, ops_perturbation_map.json, ops_stats.json
"""
import argparse
import json
import os
import sqlite3
import numpy as np
from collections import Counter


def build_db(data_dir):
    """Create ops_files.db from funk22-npy.txt."""
    txt_path = os.path.join(data_dir, 'funk22-npy.txt')
    db_path = os.path.join(data_dir, 'ops_files.db')
    assert os.path.isfile(txt_path), f"File not found: {txt_path}"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=NORMAL')
    c.execute('PRAGMA cache_size=100000')
    c.execute('DROP TABLE IF EXISTS files')
    c.execute('''
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            perturbation TEXT NOT NULL
        )
    ''')
    conn.commit()

    print("Building SQLite database from funk22-npy.txt...")
    batch = []
    batch_size = 500000
    total = 0

    with open(txt_path, 'r') as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            basename = os.path.basename(path).replace('.npy', '')
            pert = basename.split('_')[-2]
            batch.append((path, pert))
            if len(batch) >= batch_size:
                c.executemany('INSERT INTO files (file_path, perturbation) VALUES (?, ?)', batch)
                conn.commit()
                total += len(batch)
                print(f"  Inserted {total:,} rows...")
                batch = []

    if batch:
        c.executemany('INSERT INTO files (file_path, perturbation) VALUES (?, ?)', batch)
        conn.commit()
        total += len(batch)

    print(f"Creating index...")
    c.execute('CREATE INDEX idx_perturbation ON files(perturbation)')
    conn.commit()

    print(f"Database built: {total:,} rows in {db_path}")
    conn.close()


def get_perturbation_counts(db_path):
    """Return dict of perturbation -> count."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT perturbation, COUNT(*) FROM files GROUP BY perturbation')
    counts = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return counts


def get_nontargeting_paths(db_path):
    """Return list of file paths for nontargeting perturbations."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path FROM files WHERE perturbation = 'nontargeting'")
    paths = [row[0] for row in c.fetchall()]
    conn.close()
    return paths


def get_filtered_dataset(db_path, kept_genes):
    """Return (file_paths, perturbation_indices) for kept perturbations."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    placeholders = ','.join('?' for _ in kept_genes)
    c.execute(f'SELECT file_path, perturbation FROM files WHERE perturbation IN ({placeholders})', kept_genes)
    rows = c.fetchall()
    conn.close()

    gene_to_idx = {gene: i for i, gene in enumerate(kept_genes)}
    file_paths = []
    perturbation_indices = []
    for path, pert in rows:
        file_paths.append(path)
        perturbation_indices.append(gene_to_idx[pert])

    return np.array(file_paths, dtype=object), np.array(perturbation_indices, dtype=np.int32)


def compute_nontargeting_stats(nontargeting_paths, chunk_size=5000):
    """Compute per-channel mean and std from nontargeting images using batched vectorized operations."""
    n_files = len(nontargeting_paths)
    print(f"Computing stats from {n_files} nontargeting images...")

    n_channels = 4
    total_pixels = 0
    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_x2 = np.zeros(n_channels, dtype=np.float64)

    for i in range(0, n_files, chunk_size):
        chunk = nontargeting_paths[i:i + chunk_size]
        chunk_data = []
        for path in chunk:
            arr = np.load(path)
            chunk_data.append(arr)
        chunk_arr = np.stack(chunk_data).astype(np.float64)

        chunk_mean = chunk_arr.mean(axis=(0, 2, 3))
        n_pixels = chunk_arr.shape[0] * chunk_arr.shape[2] * chunk_arr.shape[3]
        sum_x += chunk_mean * n_pixels
        sum_x2 += (chunk_arr ** 2).sum(axis=(0, 2, 3))
        total_pixels += n_pixels

        if (i // chunk_size + 1) % 20 == 0 or i + chunk_size >= n_files:
            processed = min(i + chunk_size, n_files)
            print(f"  Processed {processed}/{n_files} files")

    mean = sum_x / total_pixels
    variance = (sum_x2 / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))
    return mean, std


def build_dataset_artifacts(data_dir):
    """Build ops_dataset.npz, ops_perturbation_map.json, ops_stats.json from SQLite DB."""
    db_path = os.path.join(data_dir, 'ops_files.db')
    assert os.path.isfile(db_path), f"Database not found: {db_path}. Run with --build-db first."

    # Step 1: Get perturbation counts
    print("Querying perturbation counts...")
    counts = get_perturbation_counts(db_path)
    print(f"Unique perturbations: {len(counts)}")
    print(f"Total files: {sum(counts.values()):,}")

    # Step 2: Filter perturbations
    n_nt = counts.get('nontargeting', 0)
    print(f"Nontargeting files: {n_nt:,}")

    kept_genes = sorted([
        name for name, count in counts.items()
        if name != 'nontargeting' and 4000 <= count <= 5000
    ])
    print(f"Perturbations in range [4000, 5000]: {len(kept_genes)}")

    # Step 3: Build dataset arrays
    print("Querying filtered dataset paths...")
    file_paths, perturbation_indices = get_filtered_dataset(db_path, kept_genes)
    perturbation_map = {i: gene for i, gene in enumerate(kept_genes)}
    print(f"Total samples in filtered dataset: {len(file_paths):,}")

    # Step 4: Compute nontargeting stats
    nontargeting_paths = get_nontargeting_paths(db_path)
    mean, std = compute_nontargeting_stats(nontargeting_paths)
    print(f"Per-channel mean: {mean}")
    print(f"Per-channel std:  {std}")

    # Step 5: Save artifacts
    npz_path = os.path.join(data_dir, 'ops_dataset.npz')
    np.savez(npz_path, file_paths=file_paths, perturbation_indices=perturbation_indices)
    print(f"Saved {npz_path}")

    map_path = os.path.join(data_dir, 'ops_perturbation_map.json')
    with open(map_path, 'w') as f:
        json.dump(perturbation_map, f, indent=2)
    print(f"Saved {map_path}")

    stats_path = os.path.join(data_dir, 'ops_stats.json')
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {stats_path}")

    print("Done!")


def main(args):
    data_dir = args.data_dir

    if args.build_db:
        build_db(data_dir)
        return

    build_dataset_artifacts(data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing funk22-npy.txt')
    parser.add_argument('--build-db', action='store_true',
                        help='Build SQLite database from funk22-npy.txt')
    args = parser.parse_args()
    main(args)
