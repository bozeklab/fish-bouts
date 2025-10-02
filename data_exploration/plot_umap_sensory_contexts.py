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
# reducer = umap.UMAP(random_state=42)
# embedding = reducer.fit_transform(data_nonzero)  # shape (N, 2)

# # Save the UMAP embeddings along with conditions labels
# np.savez("umap_embeddings_with_conditions.npz", 
#          embedding=embedding, 
#          labels=conditions_nonzero) 

# # Load and plot
# data = np.load("umap_embeddings_with_conditions.npz")
# embedding, conditions = data["embedding"], data["labels"]


# plt.figure(figsize=(10, 7))
# scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=conditions_nonzero, cmap="tab10", s=2, alpha=0.7)
# plt.colorbar(scatter, label="Condition")
# plt.title("UMAP projection of nonzero data")
# plt.savefig("umap_projection.png", dpi=300)

random_state = 42

def run_umap(data, n_components, random_state):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(data)
    return embedding

# # Run UMAP on non-zero PCA data
# embedding_dim_2 = run_umap(data_nonzero, n_components=2)
# embedding_dim_3 = run_umap(data_nonzero, n_components=3, random_state=random_state)

# # Save the UMAP embeddings along with motor strategies labels
# np.savez("umap_embeddings_2d_with_motor_strategies.npz", 
#          embedding=embedding_dim_2, 
#          labels=motor_strategies_data)
# print(f"Saved 2D UMAP embeddings under umap_embeddings_2d_with_motor_strategies.npz")
# np.savez("umap_embeddings_3d_with_motor_strategies.npz", 
#          embedding=embedding_dim_3, 
#          labels=conditions_nonzero)
# print(f"Saved 3D UMAP embeddings under umap_embeddings_3d_with_motor_strategies.npz")

# Plot UMAP colored by motor strategies
# For two dimensions
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

print(conditions_nonzero)

condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','Dark_Transitions(5x5cm)',
                    'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                    'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                    'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']


# For three dimensions
data = np.load("umap_embeddings_3d_with_motor_strategies.npz")
embedding, labels = data["embedding"], data["labels"]  # embedding shape: (N, 3)

# --- 3D Plot with per-class legend ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for idx, name in enumerate(condition_labels):
    mask = labels == idx
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        embedding[mask, 2],
        s=2,
        alpha=0.8,
        label=name,
    )

ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")
ax.view_init(elev=20, azim=35)  # tweak the view if you like
ax.legend(title="Sensory context", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("umap_sensory_contexts_3d.png", dpi=300)
plt.close()