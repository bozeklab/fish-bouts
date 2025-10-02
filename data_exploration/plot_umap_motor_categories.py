import numpy as np
import h5py
import matplotlib.pyplot as plt
import umap

# # Load the PCA embeddings
# with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
#     pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components
#     print(f"{pca.shape=}")  # Should be (463, 11651, 20)

# # Load the motor strategies conditions
# with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
#     motor_strategies_data = np.array(f['bout_types'])
#     print(f"{motor_strategies_data.shape=}")  # Should be (463, 11651)

# motor_strategies_names = np.load('Datasets/JM_data/classnames_jm.npy', allow_pickle=True)

# # Create a mask for non-zero data points
# nonzero_mask = np.any(pca != 0, axis=2)
# print(f"{nonzero_mask.shape=}")  # Should be (463, 11651)

# data_nonzero = pca[nonzero_mask] # shape (N, 20)
# motor_strategies_data = motor_strategies_data[nonzero_mask]  # shape (N,)

# def run_umap(data, n_components):
#     reducer = umap.UMAP(n_components=n_components, random_state=42)
#     embedding = reducer.fit_transform(data)
#     return embedding

# # Run UMAP on non-zero PCA data
# embedding_dim_2 = run_umap(data_nonzero, n_components=2)
# embedding_dim_3 = run_umap(data_nonzero, n_components=3)

# # Save the UMAP embeddings along with motor strategies labels
# np.savez("umap_embeddings_2d_with_motor_strategies.npz", 
#          embedding=embedding_dim_2, 
#          labels=motor_strategies_data)
# print(f"Saved 2D UMAP embeddings under umap_embeddings_2d_with_motor_strategies.npz")
# np.savez("umap_embeddings_3d_with_motor_strategies.npz", 
#          embedding=embedding_dim_3, 
#          labels=motor_strategies_data)
# print(f"Saved 3D UMAP embeddings under umap_embeddings_3d_with_motor_strategies.npz")

# # Plot UMAP colored by motor strategies
# # For two dimensions
# data = np.load("umap_embeddings_2d_with_motor_strategies.npz")
# embedding, labels = data["embedding"], data["labels"]

# plt.figure(figsize=(10, 7))

# for idx, name in enumerate(motor_strategies_names):
#     mask = motor_strategies_data == idx
#     plt.scatter(
#         embedding[mask, 0],
#         embedding[mask, 1],
#         s=2,
#         alpha=0.7,
#         label=name
#     )

# plt.legend(title="Motor strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.title("UMAP of PCA Embeddings colored by motor strategies")
# plt.tight_layout()
# plt.savefig("umap_motor_strategies_2d.png", dpi=300)
# plt.close()

# # For three dimensions
# data = np.load("umap3d_embeddings_with_motor_strategies.npz")
# embedding, labels = data["embedding"], data["labels"]  # embedding shape: (N, 3)

# # --- 3D Plot with per-class legend ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# for idx, name in enumerate(motor_strategies_names):
#     mask = labels == idx
#     ax.scatter(
#         embedding[mask, 0],
#         embedding[mask, 1],
#         embedding[mask, 2],
#         s=2,
#         alpha=0.8,
#         label=name,
#     )

# ax.set_xlabel("UMAP-1")
# ax.set_ylabel("UMAP-2")
# ax.set_zlabel("UMAP-3")
# ax.view_init(elev=20, azim=35)  # tweak the view if you like
# ax.legend(title="Motor strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.savefig("umap_motor_strategies_3d.png", dpi=300)
# plt.close()