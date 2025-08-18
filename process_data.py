import numpy as np
import torch
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def process_data(file_path='Datasets/JM_data/pool_ex8_PCs.h5',
                 n_components=20,
                 window_size=5,
                 save_path=None):
    """
    Load PCA data from an HDF5 file, extract non-zero bouts, create sliding windows.

    Args:
        file_path (str): Path to the HDF5 file.
        n_components (int): Number of PCA components to extract.
        window_size (int): Size of the sliding window.
        save_path (str or None): Path to save the resulting tensor. Auto-generated if None.

    Returns:
        torch.Tensor: Tensor of shape (N_windows, window_size * n_components).
    """
    all_windows = []

    with h5py.File(file_path, 'r') as f:
        pca = np.array(f['pca_fish'])[:, :, :n_components]

    nonzero_mask = np.any(pca != 0, axis=2)

    for i in range(pca.shape[0]):  # For each fish
        fish_bouts = pca[i][nonzero_mask[i]]

        if fish_bouts.shape[0] < window_size:
            continue

        for j in range(fish_bouts.shape[0] - window_size + 1):
            window = fish_bouts[j:j + window_size]
            all_windows.append(torch.tensor(window, dtype=torch.float32))


    data_tensor = torch.stack(all_windows)

    if save_path is None:
        save_dir = "processed"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"bouts_{n_components}_{window_size}.pt")

    torch.save(data_tensor, save_path)

    return data_tensor



def load_preprocessed_data(data_path,
             n_components=20,
             window_size=5):
    
    return torch.load(data_path)



def process_data_and_split(file_path='Datasets/JM_data/pool_ex8_PCs.h5',
                             conditions_idx=None,
                             n_components=20,
                             window_size=5,
                             train_ratio=0.7,
                             val_ratio=0.15,
                             test_ratio=0.15,
                             save_dir="processed_splits",
                             seed=42):
    """
    Wrapper to process data and split into train/val/test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1"
    assert conditions_idx is not None, "Must provide conditions_idx array"

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)

    all_splits = {'train': [], 'val': [], 'test': []}

    with h5py.File(file_path, 'r') as f:
        pca = np.array(f['pca_fish'])[:, :, :n_components]  # (n_fish, n_frames, n_components)
        nonzero_mask = np.any(pca != 0, axis=2)

    n_fish = len(conditions_idx)
    rng = np.random.RandomState(42)

    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(n_fish),
        train_size=0.7,
        stratify=conditions_idx,
        random_state=42
    )

    # Second split: val vs test (from temp)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,   # so val=15%, test=15%
        stratify=conditions_idx[temp_idx],
        random_state=42
    )

    print(len(train_idx), len(val_idx), len(test_idx))

    pca_train = pca[train_idx]
    pca_val   = pca[val_idx]
    pca_test  = pca[test_idx]

    X_train = make_sliding_windows(pca_train, window_size=window_size)
    X_val   = make_sliding_windows(pca_val, window_size=window_size)
    X_test  = make_sliding_windows(pca_test, window_size=window_size)


    # ---- Z-Normalization (one mean/std across all components) ----
    mean = X_train.mean()
    std = X_train.std() + 1e-8  # add eps to avoid div by zero

    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std
    # -------------------------------------------------------------

    
    all_splits['train'] = torch.tensor(X_train, dtype=torch.float32)
    all_splits['val'] = torch.tensor(X_val, dtype=torch.float32)
    all_splits['test'] = torch.tensor(X_test, dtype=torch.float32)

    for split_name, tensor in all_splits.items():
        save_path = os.path.join(save_dir, f"{split_name}_{n_components}_{window_size}.pt")
        torch.save(tensor, save_path)   

    return all_splits


def make_sliding_windows(data, window_size=5):
    """
    data: (n_fish, n_frames, n_components)
    returns: list of (n_windows, window_size, n_components)
    """
    X = []
    for fish in data:
        n_frames = fish.shape[0]
        for i in range(n_frames - window_size + 1):
            X.append(fish[i:i+window_size])
    return np.array(X)





condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','Dark_Transitions(5x5cm)',
                    'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                    'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                    'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']

condition_recs = np.array([[453,463],[121,133],[49,109],[22,49],[163,193],[109,121],
                           [133,164],[443,453],[0,22],
                           [193,258],[304,387],[258,273],[273,304],
                           [387,443]])


conditions_idx = np.full(np.max(condition_recs), -1, dtype=int)
for idx, (t0, tf) in enumerate(condition_recs):
    conditions_idx[t0:tf] = idx


