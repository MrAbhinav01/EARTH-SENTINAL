# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:38:04 2025

@author: advit
"""

import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm
import joblib
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

# -------------------------
# Supporting Classes
# -------------------------
class FastChunkedPatchLoader:
    def __init__(self, chunk_dir):
        self.chunk_dir = chunk_dir
        metadata_path = os.path.join(chunk_dir, "metadata.npy")
        self.metadata = np.load(metadata_path, allow_pickle=True).item()
        self.patches_per_chunk = self.metadata['patches_per_chunk']

    def _load_single_patch(self, patch_idx):
        chunk_idx = patch_idx // self.patches_per_chunk
        patch_in_chunk = patch_idx % self.patches_per_chunk
        chunk_file = os.path.join(self.chunk_dir, f'chunk_{chunk_idx:03d}.npy')
        if not os.path.exists(chunk_file):
            # Skip missing chunk
            return None
        chunk_data = np.load(chunk_file, mmap_mode='r')
        if patch_in_chunk >= chunk_data.shape[0]:
            return None
        patch = chunk_data[patch_in_chunk].astype(np.float32)
        return patch

    def get_patch_data(self, patch_indices):
        patches = [self._load_single_patch(idx) for idx in patch_indices]
        valid_patches = [p for p in patches if p is not None]
        return np.array(valid_patches, dtype=np.float32)

import torch.nn as nn

class FullCNN_LSTM(nn.Module):
    def __init__(self, input_channels=4, cnn_feature_dim=512, lstm_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, cnn_feature_dim)
        self.lstm = nn.LSTM(cnn_feature_dim, lstm_hidden, batch_first=True)

    def forward(self, x):
        batch_size, weeks, bands, H, W = x.shape
        cnn_out = []
        for t in range(weeks):
            xi = x[:, t]
            fi = self.cnn(xi).view(batch_size, -1)
            fi = self.fc(fi)
            cnn_out.append(fi)
        cnn_out = torch.stack(cnn_out, dim=1)
        _, (h_n, _) = self.lstm(cnn_out)
        return h_n[-1]

class Siamese_Network(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2

# -------------------------
# Inference Function
# -------------------------
def run_inference(chunk_dir="patch_chunks_final", model_path="best_model_full.pth",
                  logreg_path="logistic_regression_classifier.pkl",
                  output_tif="probability_heatmap.tif",
                  himachal_geojson="Himachal_GeoJSON.geojson"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load patch loader
    print("🔹 Loading chunked patches...")
    patch_loader = FastChunkedPatchLoader(chunk_dir)
    metadata = patch_loader.metadata
    total_patches = metadata['total_patches']
    patch_size = metadata['patch_size']
    stride = metadata['stride']
    bands = metadata['bands']
    num_weeks = metadata['num_weeks']
    H = 15516
    W = 19020
    patch_coords = [(i,j) for i in range(0,H-patch_size+1,stride)
                          for j in range(0,W-patch_size+1,stride)]
    
    # Load Siamese model
    encoder = FullCNN_LSTM(input_channels=bands)
    model = Siamese_Network(encoder).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Loaded Siamese model.")

    # Load Logistic Regression
    clf = joblib.load(logreg_path)
    print("✅ Loaded Logistic Regression classifier.")

    # Run inference
    print("🔹 Running inference on all patches...")
    all_probs = np.zeros(total_patches, dtype=np.float32)
    batch_size = 32
    for i in tqdm(range(0, total_patches, batch_size)):
        idx_batch = np.arange(i, min(i+batch_size, total_patches))
        patches = patch_loader.get_patch_data(idx_batch)
        if len(patches) == 0:
            continue
        patches_tensor = torch.tensor(patches, dtype=torch.float32).to(device)
        scores = []

        with torch.no_grad():
            # Evaluate multiple before/after splits
            for t in range(4, 11):  # t from 4 to 10 (inclusive)
                x1 = patches_tensor[:, :t]        # weeks 0 ... t-1
                x2 = patches_tensor[:, t:]        # weeks t ... 13
        
                emb1, emb2 = model(x1, x2)
                feats = torch.abs(emb1 - emb2).cpu().numpy()
                probs_t = clf.predict_proba(feats)[:, 1]
                scores.append(probs_t)

        # Final score = maximum event likelihood across temporal splits
        probs = np.max(np.stack(scores, axis=1), axis=1)
        all_probs[idx_batch[:len(probs)]] = probs


    # -------------------------
    # Reconstruct heatmap
    # -------------------------
    print("🔹 Reconstructing heatmap...")
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    for patch_idx, (i,j) in enumerate(patch_coords):
        heatmap[i:i+patch_size, j:j+patch_size] += all_probs[patch_idx]
        counts[i:i+patch_size, j:j+patch_size] += 1
    heatmap /= np.maximum(counts, 1)
    print("✅ Heatmap reconstructed.")

    # -------------------------
    # Mask with Himachal boundary
    # -------------------------
    himachal = gpd.read_file(himachal_geojson).to_crs(epsg=4326)
    # reference raster to get transform
    data_dir = "downloaded_weeks"
    ref_file = sorted([f for f in os.listdir(data_dir) if f.endswith(".tif")])[0]
    with rasterio.open(os.path.join(data_dir, ref_file)) as src:
        transform = src.transform
        raster_crs = src.crs
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    # Create mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for i_start in range(0,H-patch_size+1,stride):
        for j_start in range(0,W-patch_size+1,stride):
            ulx,uly = transform * (j_start,i_start)
            lrx,lry = transform * (j_start+patch_size, i_start+patch_size)
            patch_box = box(ulx,lry,lrx,uly)
            if himachal.geometry.intersects(patch_box).any():
                mask[i_start:i_start+patch_size, j_start:j_start+patch_size] = 1
    heatmap *= mask

    # -------------------------
    # Save GeoTIFF
    # -------------------------
    output_path = output_tif
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(heatmap, 1)
    print(f"🎉 Probability heatmap saved as {output_path}!")

    # -------------------------
    # Save PNG for visualization
    # -------------------------
    plt.figure(figsize=(10,10))
    min_val, max_val = np.min(heatmap), np.max(heatmap)
    plt.imshow((heatmap - min_val) / (max_val - min_val), cmap='hot')
    plt.colorbar(label="Normalized Risk")

    plt.title("Himachal Risk Heatmap")
    plt.axis("off")
    plt.savefig("himachal_risk_map.png", dpi=300)
    plt.show()
    print("🎉 PNG visualization saved as himachal_risk_map.png")

if __name__ == "__main__":
    run_inference()
