import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch


def cosine_similarity(a, b):
    print(f"Shapes: a={a.shape}, b={b.shape}")
    dot = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    return dot / (norm_a * norm_b)

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components

valid_mask = np.any(pca != 0, axis=-1)  # True where bouts have data

true_means = []
shuffled_means = []

for fish_idx in range(pca.shape[0]):
    fish_data = pca[fish_idx]      # (11651, 20)
    valid = valid_mask[fish_idx]   # (11651,)

    valid_indices = np.where(valid)[0]
    if len(valid_indices) < 2:
        true_means.append(np.nan)
        shuffled_means.append(np.nan)
        continue

    # --- True neighbor similarity ---
    sims_true = cosine_similarity(
        fish_data[valid_indices[:-1]],
        fish_data[valid_indices[1:]]
    )

    # --- Shuffled neighbor similarity ---
    shuffled = fish_data[valid_indices].copy()
    np.random.shuffle(shuffled)  # in-place shuffle along bout axis
    sims_shuf = cosine_similarity(
        shuffled[:-1],
        shuffled[1:]
    )

    true_means.append(np.mean(sims_true))
    shuffled_means.append(np.mean(sims_shuf))

# Convert to arrays
true_means = np.array(true_means)
shuffled_means = np.array(shuffled_means)


# Plot comparison
plt.figure(figsize=(8,5))
plt.scatter(shuffled_means, true_means, s=8, alpha=0.3, color='steelblue')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("Mean cosine similarity (shuffled order)")
plt.ylabel("Mean cosine similarity (true order)")
plt.title("True vs Shuffled Bout Neighbor Similarity per Fish")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig("bout_similarity_comparison.png", dpi=300, bbox_inches='tight')

# Print summary
print(f"Mean (true): {np.nanmean(true_means):.4f}")
print(f"Mean (shuffled): {np.nanmean(shuffled_means):.4f}")
print(f"Mean difference: {np.nanmean(true_means - shuffled_means):.4f}")


print(f"{len(true_means)=}, {np.sum(np.isnan(true_means))=}, {np.sum(np.isnan(shuffled_means))=}")

best_fish_idx = np.nanargmax(true_means)
best_fish_value = true_means[best_fish_idx]

print(f"Fish index with highest mean cosine similarity: {best_fish_idx}")
print(f"Mean cosine similarity: {best_fish_value:.4f}")

fish_data = pca[best_fish_idx]          # (11651, 20)
valid = valid_mask[best_fish_idx]       # (11651,)
valid_indices = np.where(valid)[0]

# Compute consecutive cosine similarities
sims_seq = cosine_similarity(
    fish_data[valid_indices[:-1]],
    fish_data[valid_indices[1:]]
)


plt.figure(figsize=(30,4))
plt.plot(range(len(sims_seq)), sims_seq, marker='o', markersize=3, alpha=0.7)
plt.xlabel("Bout index (in sequence)")
plt.ylabel("Cosine similarity to next bout")
plt.title(f"Consecutive Bout Cosine Similarities — Fish {best_fish_idx}")
plt.grid(alpha=0.3)
plt.savefig("consecutive_bout_similarity.png", dpi=300, bbox_inches='tight')    

# Find fish with lowest mean cosine similarity (ignoring NaNs)
worst_fish_idx = np.nanargmin(true_means)
print(f"Worst fish index: {worst_fish_idx}, mean similarity: {true_means[worst_fish_idx]:.4f}")

# Extract its valid bouts
fish_data = pca[worst_fish_idx]         # (11651, 20)
valid = valid_mask[worst_fish_idx]      # (11651,)
valid_indices = np.where(valid)[0]

# Compute consecutive cosine similarities
sims_seq = cosine_similarity(
    fish_data[valid_indices[:-1]],
    fish_data[valid_indices[1:]]
)

# Plot the sequence
plt.figure(figsize=(30,4))
plt.plot(range(len(sims_seq)), sims_seq, marker='o', markersize=3, alpha=0.7, color='darkred')
plt.xlabel("Bout index (in sequence)")
plt.ylabel("Cosine similarity to next bout")
plt.title(f"Consecutive Bout Cosine Similarities — Fish {worst_fish_idx}")
plt.grid(alpha=0.3)
plt.savefig("worst_fish_bout_similarity.png", dpi=300, bbox_inches='tight')
