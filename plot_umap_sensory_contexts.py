import numpy as np
import h5py
import matplotlib.pyplot as plt
import umap

# Load the PCA embeddings
with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components

# Create a mask for non-zero data points
nonzero_mask = np.any(pca != 0, axis=2)

print(f"{pca.shape=}")  # Should be (463, 11651, 20)
print(f"{nonzero_mask.shape=}")  # Should be (463, 11651)

# Load the sensory contexts conditions
conditions_idx = np.load('sensory_contexts_data.npy')
print(f"{conditions_idx.shape=}")


# Extend conditions
conditions_expanded = np.repeat(conditions_idx[:, np.newaxis], pca.shape[1], axis=1)  # shape (463, 11651)

data_nonzero = pca[nonzero_mask]              # shape (N, 20)
conditions_nonzero = conditions_expanded[nonzero_mask]  # shape (N,)

# Flattening already handled by masking (data is now 2D, conditions 1D)
print(data_nonzero.shape, conditions_nonzero.shape)

# Run UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data_nonzero)  # shape (N, 2)

# Save the UMAP embeddings along with conditions labels
np.savez("umap_embeddings_with_conditions.npz", 
         embedding=embedding, 
         labels=conditions_nonzero) 

# Load and plot
data = np.load("umap_embeddings_with_conditions.npz")
embedding, conditions = data["embedding"], data["labels"]


plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=conditions_nonzero, cmap="tab10", s=2, alpha=0.7)
plt.colorbar(scatter, label="Condition")
plt.title("UMAP projection of nonzero data")
plt.savefig("umap_projection.png", dpi=300)
