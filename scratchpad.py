import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch

# ['MetaData', 'cov', 'data_means', 'eigvecs', 'max_shuffs', 'pca_fish', 'seeds', 'var_exp']

file_path = 'Datasets/JM_data/pool_ex8_PCs.h5'
with h5py.File(file_path, 'r') as f:
    print(list(f.keys()))  # List top-level groups/datasets

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    data = f['MetaData']
    print(type(data))           # Check if it's a group or dataset
    print(list(data.keys()))    # If it's a group, list its contents

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    lengths_data = f['MetaData/lengths_data'][:]
    print(lengths_data.shape)
    print(lengths_data)

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    data = f['var_exp']
    print(type(data))
    print(data.shape)
    print(data[:])

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    print(f"{f['cov'].shape=}") 
    print(f"{f['data_means'].shape=}")  
    print(f"{f['eigvecs'].shape=}") 
    print(f"{f['max_shuffs'].shape=}")  
    print(f"{f['pca_fish'].shape=}")  
    print(f"{f['seeds'].shape=}") 
    print(f"{f['var_exp'].shape=}") 

with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components
    print(f"{pca.shape=}")  # Should be (463, 11651, 20)

total_nonzero_bouts = np.sum(np.any(pca != 0, axis=2))
print(f"{total_nonzero_bouts=}")

nonzero_mask = np.any(pca != 0, axis=2)  # shape: (463, 11651)

# Extract non-zero bouts (up to 20 per fish) and stack
nonzero_bouts = np.vstack([
    pca[i][nonzero_mask[i]] for i in range(pca.shape[0])
])  # shape:

print(f"{nonzero_bouts.shape=}")  # Should be (≤ 463×20, 20)



window_size = 5
all_windows = []

for i in range(pca.shape[0]):  # For each fish
    fish_bouts = pca[i][nonzero_mask[i]]  # Shape: (num_bouts_i, 20)
    
    if fish_bouts.shape[0] < window_size:
        continue  # Skip if not enough bouts

    # Sliding window
    for j in range(fish_bouts.shape[0] - window_size + 1):
        window = fish_bouts[j:j + window_size]  # Shape: (5, 20)
        all_windows.append(torch.tensor(window, dtype=torch.float32))

# Final dataset
input_sequences = torch.stack(all_windows)  # Shape: (N_windows, 5, 20)


# Compute statistics
mean = np.mean(nonzero_bouts, axis=0)
std = np.std(nonzero_bouts, axis=0)
min = np.min(nonzero_bouts, axis=0)
max = np.max(nonzero_bouts, axis=0)

overall_mean = np.mean(nonzero_bouts)
overall_std = np.std(nonzero_bouts)

print(f"Mean per PCA component: {mean}")
print(f"Standard deviation per PCA component: {std}")
print(f"Min per PCA component: {min}")
print(f"Max per PCA component: {max}")
print(f"Overall mean: {overall_mean}")
print(f"Overall standard deviation: {overall_std}")

nonzero_counts = np.sum(nonzero_mask, axis=1)
print(f"Min non-zero bouts per fish: {np.min(nonzero_counts)}")
print(f"Max non-zero bouts per fish: {np.max(nonzero_counts)}")

# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
# bouts_2d = reducer.fit_transform(nonzero_bouts)

# Plot
# plt.scatter(bouts_2d[:, 0], bouts_2d[:, 1], s=2, alpha=0.5)
# plt.title("UMAP of Non-Zero Bout Embeddings")
# plt.xlabel("UMAP-1")
# plt.ylabel("UMAP-2")
# plt.savefig("umap_plot.png", dpi=300, bbox_inches='tight')
# plt.show()



# tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
# bouts_2d = tsne.fit_transform(nonzero_bouts)

# plt.scatter(bouts_2d[:, 0], bouts_2d[:, 1], s=2, alpha=0.5)
# plt.title("t-SNE of Non-Zero Bout Embeddings")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.show()
