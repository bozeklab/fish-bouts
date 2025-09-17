import h5py
import numpy as np
import numpy.ma as ma # masked arrays
import matplotlib.pyplot as plt
import umap

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = ma.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components
    print(f"{pca.shape=}")  # Should be (463, 11651, 20)

nonzero_mask = np.any(pca != 0, axis=2)
data_nonzero = pca[nonzero_mask] # shape (N, 20)
print(f"{nonzero_mask.shape=}")  # Should be (463, 11651)
print(f"{data_nonzero.shape=}")  # Should be (N, 20)

with h5py.File("Datasets/JM_data/filtered_jmpool_kin.h5") as f:
    print(list(f.keys()))
    orientations = ma.array(f["orientation_smooth"])
    print(f"{orientations.shape=}")  # (463, 11651, 175)
    print(f"Orientation range before masking: {orientations.min()} to {orientations.max()}")
    # orientations_averaged_per_bout = ma.mean(orientations, axis=-1) 
    # print(f"{orientations_averaged_per_bout.shape=}")  # (463, 11651)

# orientations_nonzero = orientations[nonzero_mask]  # shape (N, 175)
# print(f"{orientations_nonzero.shape=}")
# print(f"Orientation range before masking: {ma.min(orientations)} to {ma.max(orientations)}")

# MY KINEMATICS CALCULATION
mask = orientations != 0 
print(f"{mask.shape=}")
idx = mask.cumsum(axis=-1) 
print(f"{idx.shape=}")
print(f"{idx[0, 0, :]=}")
last_idx = idx.argmax(axis=-1)

B, T, L = orientations.shape
bt_idx = np.indices((B, T))
orientations_last_values = orientations[bt_idx[0], bt_idx[1], last_idx]  # shape (B,T)

del_orientations = orientations[:, :, 0] - orientations_last_values
print(f"Orientation range after masking: {del_orientations[:5]=}")

print(f"{del_orientations.shape=}")
del_orientations_degrees = del_orientations * (180 / np.pi)




del_orientations_degrees_non_zero = del_orientations_degrees[nonzero_mask]  # shape (N,)
print(f"{del_orientations_degrees_non_zero.shape=}")
# mask = orientations.mask 
# n_fish, n_bouts, n_frames = orientations.shape

# # Build frame indices
# frame_idx = np.arange(n_frames)
# frame_idx = np.broadcast_to(frame_idx, (n_fish, n_bouts, n_frames))
# print(f"{frame_idx.shape=}")

# # Replace masked positions with -1 so they don’t count
# valid_idx = np.where(~mask, frame_idx, -1)

# # Take max along frames → last valid index for each (fish, bout)
# last_non_masked_idx = valid_idx.max(axis=2)  # shape (n_fish, n_bouts)

# # Now gather the values at those indices
# fish_idx, bout_idx = np.meshgrid(np.arange(n_fish), np.arange(n_bouts), indexing="ij")

# last_orientations = orientations[fish_idx, bout_idx, last_non_masked_idx]

# # Compute difference between first and last orientation
# del_orientations = orientations[:, :, 0] - last_orientations
# print(f"{del_orientations.shape=}")
# del_orientations_degrees = del_orientations * (180 / np.pi)
# del_orientations = orientations[:, :, 0] - orientations[:, :, last_non_masked_idx]  # change in orientation between bouts
# del_orientations_degrees = del_orientations * (180 / np.pi)
# headings = ma.zeros(orientations.shape)
# headings[:,:-1,:] = del_orientations_degrees
# headings[:,-1,:] = ma.masked

########################################


# data_nonzero = pca[nonzero_mask] # shape (N, 20)
# orientations = orientations_averaged_per_bout[nonzero_mask]  # shape (N,)
# print(f"{data_nonzero.shape=}")
# print(f"{orientations.shape=}")
# print(f"Orientation range: {orientations.min()} to {orientations.max()}")
# print(f"{orientations[400:900]=}")
seed = 57
def run_umap(data, n_components, seed):
    reducer = umap.UMAP(n_components=n_components, random_state=seed)
    embedding = reducer.fit_transform(data)
    return embedding

# Run UMAP on non-zero PCA data
embedding_dim_2 = run_umap(data_nonzero, n_components=2, seed=seed)
embedding_dim_3 = run_umap(data_nonzero, n_components=3, seed=seed)

# Save the UMAP embeddings along with motor strategies labels
np.savez("umap_embeddings_2d_heading.npz", 
         embedding=embedding_dim_2, 
         labels=del_orientations_degrees_non_zero)
print(f"Saved 2D UMAP embeddings under umap_embeddings_2d_heading.npz")
np.savez("umap_embeddings_3d_heading.npz", 
         embedding=embedding_dim_3, 
         labels=del_orientations_degrees_non_zero)
print(f"Saved 3D UMAP embeddings under umap_embeddings_3d_heading.npz")
del_orientations_degrees_non_zero = (del_orientations_degrees_non_zero + 180) % 360 - 180
#del_orientations_degrees_non_zero = np.clip(del_orientations_degrees_non_zero, -90, 90)

# Plot UMAP colored by motor strategies
# For two dimensions
data = np.load("umap_embeddings_2d_heading.npz")
embedding, labels = data["embedding"], data["labels"]
print(f"{embedding.shape=}, {labels.shape=}")
plt.figure(figsize=(8,6))
sc = plt.scatter(
    embedding[:,0], embedding[:,1], 
    c=del_orientations_degrees_non_zero, cmap="viridis", s=1, alpha=0.2
)
plt.colorbar(sc, label="Orientation")
plt.title("UMAP colored by orientation")
plt.savefig(f"umap_2d_heading_{seed}.png", dpi=300)
plt.show()
