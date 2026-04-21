# -*- coding: utf-8 -*-
"""
Train Logistic Regression on Siamese Embeddings
After training Siamese Network (best_model_full.pth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,f1_score,accuracy_score,precision_score,recall_score
import joblib
import os

RESULTS_PATH = "logreg_results.txt"

# ---------------- Load Supporting Classes ----------------
class FastChunkedPatchLoader:
    def __init__(self, chunk_dir):
        self.chunk_dir = chunk_dir
        metadata_path = os.path.join(chunk_dir, "metadata.npy")
        self.metadata = np.load(metadata_path, allow_pickle=True).item()

    def _load_single_patch_direct(self, patch_idx):
        patches_per_chunk = self.metadata['patches_per_chunk']
        chunk_idx = patch_idx // patches_per_chunk
        patch_in_chunk = patch_idx % patches_per_chunk
        chunk_file = os.path.join(self.chunk_dir, f'chunk_{chunk_idx:03d}.npy')

        chunk_data = np.load(chunk_file, mmap_mode='r')
        patch = chunk_data[patch_in_chunk].astype(np.float32)

        patch = np.clip(patch, -30000, 30000)
        for b in range(patch.shape[0]):
            band = patch[b]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                patch[b] = (band - band_min) / (band_max - band_min)
            else:
                patch[b] = 0
        return patch

    def get_patch_data(self, patch_indices):
        patches = [self._load_single_patch_direct(idx) for idx in patch_indices]
        return np.array(patches, dtype=np.float32)


# ---------------- Model Definitions ----------------
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


# ---------------- Dataset ----------------
class FastPatchPairsDataset(Dataset):
    def __init__(self, patches, pairs, labels):
        self.patches = patches
        self.pairs = pairs
        self.labels = labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        a_idx, b_idx = self.pairs[idx]
        x1 = torch.tensor(self.patches.get_patch_data([a_idx]), dtype=torch.float32)[0]
        x2 = torch.tensor(self.patches.get_patch_data([b_idx]), dtype=torch.float32)[0]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, y


# ---------------- Main Script ----------------
if __name__ == "__main__":
    print("🚀 Logistic Regression Training on Siamese Embeddings")

    # Load patch and pair data
    CHUNK_DIR = "patch_chunks"
    pairs = np.load("siamese_week_pairs.npy")
    labels = np.load("siamese_week_pair_labels.npy")

    from sklearn.model_selection import train_test_split
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    patch_loader = FastChunkedPatchLoader(CHUNK_DIR)
    train_dataset = FastPatchPairsDataset(patch_loader, train_pairs, train_labels)
    test_dataset = FastPatchPairsDataset(patch_loader, test_pairs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load trained Siamese model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FullCNN_LSTM(input_channels=patch_loader.metadata['bands'])
    model = Siamese_Network(encoder).to(device)
    checkpoint = torch.load("best_model_full.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✅ Loaded pretrained Siamese model!")

    # ---------------- Extract Features ----------------
    def extract_features(loader, model, device):
        feats, lbls = [], []
        with torch.no_grad():
            for x1, x2, y in tqdm(loader, desc="Extracting embeddings"):
                x1, x2 = x1.to(device), x2.to(device)
                emb1, emb2 = model(x1, x2)
                feat = torch.abs(emb1 - emb2).cpu().numpy()
                feats.append(feat)
                lbls.extend(y.cpu().numpy())
        return np.vstack(feats), np.array(lbls)

    print("\n🔍 Generating embeddings for train/test sets...")
    train_X, train_y = extract_features(train_loader, model, device)
    test_X, test_y = extract_features(test_loader, model, device)
    print(f"Feature shape: {train_X.shape}")

    # ---------------- Train Logistic Regression ----------------
    print("\n⚙️ Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(train_X, train_y)

    print("\n🔍 Evaluating on test data...")
    pred_y = clf.predict(test_X)
    
    acc = accuracy_score(test_y, pred_y)
    prec = precision_score(test_y, pred_y)
    rec = recall_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y)
    cm = confusion_matrix(test_y, pred_y)
    
    # ------------------------------------------
    # 8️⃣ Print results neatly
    # ------------------------------------------
    print("\n📊 Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:\n")
    print(classification_report(test_y, pred_y, digits=3))
    
    # ------------------------------------------
    # 9️⃣ Save results to file
    # ------------------------------------------
    print(f"\n💾 Saving results to {RESULTS_PATH} ...")
    with open(RESULTS_PATH, "w") as f:
        f.write("Siamese + Logistic Regression Evaluation Results\n")
        f.write("===============================================\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(test_y, pred_y, digits=3))
    
    # ------------------------------------------
    # 🔟 Save classifier for future reuse
    # ------------------------------------------
    joblib.dump(clf, "logistic_regression_classifier.pkl")
    print("\n✅ Logistic Regression model saved as logistic_regression_classifier.pkl")
    print("✅ Metrics saved to logreg_results.txt")
    print("\n🎉 Done! Everything completed successfully.\n")