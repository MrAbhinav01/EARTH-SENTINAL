# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:03:30 2025

@author: advit
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def create_chunked_numpy_arrays():
    DATA_DIR = "downloaded_weeks"
    PATCH_SIZE = 256
    STRIDE = 256
    OUT_DIR = "patch_chunks_final"
    PATCHES_PER_CHUNK = 500

    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".tif")])
    num_weeks = len(files)

    # Open first TIFF to get dimensions and bands
    with rasterio.open(os.path.join(DATA_DIR, files[0])) as src:
        bands, H, W = src.count, src.height, src.width

    num_patches_h = (H - PATCH_SIZE) // STRIDE + 1
    num_patches_w = (W - PATCH_SIZE) // STRIDE + 1
    num_patches = num_patches_h * num_patches_w
    num_chunks = (num_patches + PATCHES_PER_CHUNK - 1) // PATCHES_PER_CHUNK

    metadata = {
        'patch_size': PATCH_SIZE,
        'stride': STRIDE,
        'bands': bands,
        'num_weeks': num_weeks,
        'total_patches': num_patches,
        'patches_per_chunk': PATCHES_PER_CHUNK,
        'num_chunks': num_chunks
    }
    np.save(os.path.join(OUT_DIR, 'metadata.npy'), metadata)

    patch_idx = 0
    chunk_idx = 0
    current_chunk_patches = []

    for i in range(0, H - PATCH_SIZE + 1, STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch_data = np.zeros((num_weeks, bands, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            skip_patch = False

            for week_idx, file_name in enumerate(files):
                tiff_path = os.path.join(DATA_DIR, file_name)
                try:
                    with rasterio.open(tiff_path) as src:
                        patch = src.read(window=rasterio.windows.Window(j, i, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
                        patch_data[week_idx] = patch
                except Exception as e:
                    print(f"⚠️ Skipping corrupt patch at row {i}, col {j} in file {file_name}: {e}")
                    skip_patch = True
                    break  # skip this patch entirely if any week fails

            if skip_patch:
                continue

            current_chunk_patches.append(patch_data)
            patch_idx += 1

            

            # Save chunk
            if len(current_chunk_patches) == PATCHES_PER_CHUNK or patch_idx == num_patches:
                chunk_file = os.path.join(OUT_DIR, f'chunk_{chunk_idx:03d}.npy')
                if os.path.exists(chunk_file):
                    print(f"✅ Skipping already saved chunk {chunk_idx}")
                else:
                    chunk_array = np.array(current_chunk_patches, dtype=np.float32)
                    np.save(chunk_file, chunk_array)
                    print(f"Saved chunk {chunk_idx} with {len(current_chunk_patches)} patches")
                    print(f"Week {week_idx}, File {file_name}, Band stats: min={patch.min()}, max={patch.max()}, mean={patch.mean():.2f}")
                current_chunk_patches = []
                chunk_idx += 1

            if patch_idx % 50 == 0:
                print(f"🟢 Processed {patch_idx}/{num_patches} patches...")

    print(f"🎉 Created {chunk_idx} chunk files!")

if __name__ == "__main__":
    create_chunked_numpy_arrays()
