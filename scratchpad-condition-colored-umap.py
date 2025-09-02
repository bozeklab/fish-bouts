import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch



conditions_idx = np.load('sensory_contexts_data.npy')
print(f"{conditions_idx.shape=}")

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components

print(f"{pca.shape=}")  # Should be (463, 11651, 20)
nonzero_mask = np.any(pca != 0, axis=2)
print(f"{nonzero_mask.shape=}")  # Should be (463, 11651)


# Step 1: Extend conditions
conditions_expanded = np.repeat(conditions_idx[:, np.newaxis], pca.shape[1], axis=1)  # shape (463, 11651)

# Step 2: Apply mask
data_nonzero = pca[nonzero_mask]              # shape (N, 20)
conditions_nonzero = conditions_expanded[nonzero_mask]  # shape (N,)

# Step 3: Flattening already handled by masking (data is now 2D, conditions 1D)
print(data_nonzero.shape, conditions_nonzero.shape)

# Step 4: Run UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data_nonzero)  # shape (N, 2)

# Step 5: Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=conditions_nonzero, cmap="tab10", s=2, alpha=0.7)
plt.colorbar(scatter, label="Condition")
plt.title("UMAP projection of nonzero data")
plt.savefig("umap_projection.png", dpi=300)
