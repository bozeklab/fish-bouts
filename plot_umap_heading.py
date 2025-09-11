import h5py
import numpy as np
import matplotlib.pyplot as plt
import umap

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components
    print(f"{pca.shape=}")  # Should be (463, 11651, 20)


with h5py.File("Datasets/JM_data/filtered_jmpool_kin.h5") as f:
    print(list(f.keys()))
    orientations = np.array(f["orientation_smooth"])
    print(f"{orientations.shape=}")  # Should be (463, 11651)
    print(f"Orientation range: {orientations.min()} to {orientations.max()}")
    orientations_averaged_per_bout = np.mean(orientations, axis=-1) 
    print(f"{orientations_averaged_per_bout.shape=}")  # Should be (463, 11651)

nonzero_mask = np.any(pca != 0, axis=2)
print(f"{nonzero_mask.shape=}")  # Should be (463, 11651)

data_nonzero = pca[nonzero_mask] # shape (N, 20)
orientations = orientations_averaged_per_bout[nonzero_mask]  # shape (N,)
print(f"{data_nonzero.shape=}")
print(f"{orientations.shape=}")
print(f"Orientation range: {orientations.min()} to {orientations.max()}")
print(f"{orientations[400:900]=}")

def run_umap(data, n_components):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(data)
    return embedding

# Run UMAP on non-zero PCA data
embedding_dim_2 = run_umap(data_nonzero, n_components=2)
embedding_dim_3 = run_umap(data_nonzero, n_components=3)

# Save the UMAP embeddings along with motor strategies labels
np.savez("umap_embeddings_2d_with_motor_strategies.npz", 
         embedding=embedding_dim_2, 
         labels=orientations)
print(f"Saved 2D UMAP embeddings under umap_embeddings_2d_with_motor_strategies.npz")
np.savez("umap_embeddings_3d_with_motor_strategies.npz", 
         embedding=embedding_dim_3, 
         labels=orientations)
print(f"Saved 3D UMAP embeddings under umap_embeddings_3d_with_motor_strategies.npz")

# Plot UMAP colored by motor strategies
# For two dimensions
data = np.load("umap_embeddings_2d_with_motor_strategies.npz")
embedding, labels = data["embedding"], data["labels"]

plt.figure(figsize=(8,6))
sc = plt.scatter(
    embedding[:,0], embedding[:,1], 
    c=labels, cmap="viridis", s=20
)
plt.colorbar(sc, label="Orientation")
plt.title("UMAP colored by orientation")
plt.show()
