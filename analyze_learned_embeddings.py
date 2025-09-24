import torch
import torch.nn.functional as F
import yaml
from process_data import one_hot_process_data_and_split
from models.encoder import TransformerEncoder
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import h5py
from process_data import labels_to_onehot
from process_data import make_sliding_windows

def analyze_learned_embeddings(model, input_embeddings):
    model.eval()
    with torch.no_grad():
        preds, learned_embeddings = model(input_embeddings, return_embeddings=True)

    return learned_embeddings


K=10
B=30

# Load the PCA embeddings
with h5py.File('Datasets/JM_data/pool_ex8_PCs.h5', 'r') as f:
    pca = np.array(f['pca_fish'])[:, :, :20]  # Extract only first 20 PCA components
    print(f"{pca.shape=}")  # Should be (463, 11651, 20)

# Load the motor strategies conditions
with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
    motor_strategies_data = np.array(f['bout_types'])
    print(f"{motor_strategies_data.shape=}")  # Should be (463, 11651)

# Create a mask for non-zero data points
nonzero_mask = np.any(pca != 0, axis=2)
print(f"{nonzero_mask.shape=}")  # Should be (463, 11651)

# data_nonzero = pca[nonzero_mask] # shape (N, 20)
# motor_strategies_data = motor_strategies_data[nonzero_mask]  # shape (N,)

onehot_all = labels_to_onehot(motor_strategies_data, num_classes=13).astype(np.float32)  # (n_fish, n_frames, 13)
onehot_filtered = onehot_all[nonzero_mask]

onehot_windows = make_sliding_windows(onehot_filtered, window_size=K)

print(f"{onehot_windows.shape=}")  # Should be (N_windows, K, 13)

input_batch = torch.tensor(onehot_windows[:B])

print(f"{input_batch.shape=}")  # Should be (B, K, 13)
labeling_matrix = torch.arange(K).unsqueeze(0) + torch.arange(B).unsqueeze(1)
print(f"{labeling_matrix.shape=}")  # (K, B)
print(f"{labeling_matrix=}")






#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# # better not to use the data processing, because there is some randomness
# # instead, we process the data directly here
# labels_path = 'Datasets/JM_data/filtered_jmpool_kin.h5'
# with h5py.File(labels_path, 'r') as f:
#     motor_strategies = np.array(f['bout_types'])  # expected shape (n_fish, n_frames), ints 0..12

# onehot_all = labels_to_onehot(motor_strategies, num_classes=13).astype(np.float32)  # (n_fish, n_frames, 13)



model = TransformerEncoder(
    input_dim=input_batch.shape[-1],
    seq_len=config["data"]["sequence_length"],
    d_model=config["model"]["d_model"],
    nhead=config["model"]["nhead"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"],
    learnable_mask_embedding=config["masking"]["learnable_mask_embedding"]
).to(device)


ckpt = torch.load("best_model.pth", map_location="cpu")
print(ckpt.keys())
model.load_state_dict(ckpt['model_state_dict'], strict=True)

model.eval()


learned_embeddings = analyze_learned_embeddings(model, input_batch)
print(f"{learned_embeddings.shape=}")

X = learned_embeddings.detach().cpu().numpy().reshape(-1, learned_embeddings.shape[-1])  # (B*10, 256)
y = (labeling_matrix.detach().cpu().numpy() 
        if isinstance(labeling_matrix, torch.Tensor) else np.asarray(labeling_matrix)).reshape(-1)  # (B*10,)



# PCA to 2D
pca = PCA(n_components=2, random_state=0)
X2 = pca.fit_transform(X)

# plot, one color per class with legend
plt.figure(figsize=(7, 6))
classes = np.unique(y)
for c in classes:
    pts = X2[y == c]
    plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.8, label=str(c))
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(title="Label", markerscale=1.5, frameon=False)
plt.tight_layout()
plt.savefig("pca_learned_embeddingsCOLORED.png", dpi=300)

# idx = torch.arange(N_SEQUENCES)
# # picked = embeddings[idx, N_SEQUENCES -1 - idx, :]

# picked = embeddings[:, 0, :]

# print(f"{picked.shape=}")

# pca = PCA(n_components=2, random_state=0)
# X2 = pca.fit_transform(picked.cpu().numpy())

# # Plot
# plt.figure(figsize=(5, 4))
# plt.scatter(X2[:, 0], X2[:, 1])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA of matched embeddings across windows")
# plt.tight_layout()
# plt.savefig("pca_matched_embeddings.png", dpi=300)

# print("Explained variance ratio:", pca.explained_variance_ratio_)


# X = F.normalize(picked, p=2, dim=1, eps=1e-12)     # row-normalize
# S = X @ X.T                                        # (10, 10), values in [-1, 1]

# print(S.shape)  # torch.Size([10, 10])

# # Optional: convert to "cosine distance"
# D = 1 - S

# # Plot heatmap
# S_np = S.detach().cpu().numpy()
# plt.figure(figsize=(5, 4))
# im = plt.imshow(S_np, vmin=-1, vmax=1, aspect="equal")
# plt.colorbar(im, label="cosine similarity")
# ticks = np.arange(10)
# plt.xticks(ticks, ticks + 1)   # label windows 1..10
# plt.yticks(ticks, ticks + 1)
# plt.title("Pairwise cosine similarity of selected embeddings")
# plt.xlabel("Window")
# plt.ylabel("Window")
# plt.tight_layout()
# plt.savefig("cosine_similarity_matched_embeddings.png", dpi=300)