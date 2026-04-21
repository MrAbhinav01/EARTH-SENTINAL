# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 09:51:52 2025

@author: advit
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Replace this with your confusion matrix ===
cm = np.array([[1986,  814],
               [ 103, 2697]])

# Class names (0 = No Landslide, 1 = Landslide)
labels = ["No Landslide", "Landslide"]

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.title("Confusion Matrix (Siamese Embeddings + Logistic Regression)", fontsize=16)

plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=300)
plt.show()

print("✅ Heatmap saved as: confusion_matrix_heatmap.png")
