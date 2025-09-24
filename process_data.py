import numpy as np
import torch
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import compute_statistics

def make_sliding_windows(data, window_size):
    print(f"Creating sliding windows of size {window_size}...")
    return np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))[:, 0, :, :]


def filter_nonzero(pca, mask):
    """
    pca: (n_fish, n_frames, n_components)
    mask: (n_fish, n_frames) boolean
    returns: (total_valid_frames, n_components)
    """
    filtered = pca[mask]
    return filtered


def labels_to_onehot(labels, num_classes=13, return_valid_mask=False):
    """
    labels: array-like (...,) possibly float/str; values should be in [0, num_classes-1]
    return_valid_mask: if True, also returns a boolean mask of valid indices
    """
    lab = np.asarray(labels)

    # Coerce to integers robustly
    if lab.dtype.kind in ('U', 'S', 'O'):  # strings/bytes/object
        lab = lab.astype(np.int64)
    elif lab.dtype.kind == 'f':            # floats
        if not np.all(np.isfinite(lab)):
            raise ValueError("Labels contain non-finite values (NaN/Inf).")
        lab = np.rint(lab).astype(np.int64)
    elif lab.dtype.kind in ('i', 'u'):     # already integer/unsigned
        lab = lab.astype(np.int64)
    else:
        raise TypeError(f"Unsupported labels dtype: {lab.dtype}")

    lab = lab - 1
    # Build one-hot with validity check
    valid = (lab >= 0) & (lab < num_classes)
    eye = np.eye(num_classes, dtype=np.float32)
    onehot = np.zeros(lab.shape + (num_classes,), dtype=np.float32)
    onehot[valid] = eye[lab[valid]]

    if return_valid_mask:
        return onehot, valid
    return onehot


def process_data_and_split(config):
    """
    Process raw data, split into train/val/test, apply sliding window, and save processed data.
    """
    file_path = config["data"]["raw_path"]
    conditions_path = config["data"]["conditions_path"]
    n_components = config["data"]["input_dim"]
    window_size = config["data"]["sequence_length"]
    train_fraction = config["training"]["train_fraction"]
    val_fraction = config["training"]["val_fraction"]
    processed_dir = config["data"]["processed_dir"]
    seed = config["seed"]


    with h5py.File(file_path, 'r') as f:
        pca = np.array(f['pca_fish'])[:, :, :n_components]  # (n_fish, n_frames, n_components)
        nonzero_mask = np.any(pca != 0, axis=2)

    conditions_idx = np.load(conditions_path)
    n_fish = len(conditions_idx)

    print(f"Total fish: {n_fish}, PCA shape: {pca.shape}, Nonzero mask shape: {nonzero_mask.shape}")
    print(f"Conditions index shape: {conditions_idx.shape}")
    
    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(n_fish),
        train_size=train_fraction,
        stratify=conditions_idx,
        random_state=seed
    )

    print(f"Train size: {len(train_idx)}, Val+Test size: {len(temp_idx)}")

    # Second split: val vs test (from temp)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_fraction / (1 - train_fraction),
        stratify=conditions_idx[temp_idx],
        random_state=seed
    ) 
    print(f"Validation size: {len(val_idx)}, Test size: {len(test_idx)}")

    # Filter out zero frames
    pca_train = filter_nonzero(pca[train_idx], nonzero_mask[train_idx])
    pca_val   = filter_nonzero(pca[val_idx],   nonzero_mask[val_idx])
    pca_test  = filter_nonzero(pca[test_idx],  nonzero_mask[test_idx])

    print(f"Filtered PCA shapes before sliding windowing: train={len(pca_train)}, val={len(pca_val)}, test={len(pca_test)}")
    print("------------------------------------------------")

    print("pca_train", pca_train.shape)
    print("pca_val", pca_val.shape)
    print("pca_test", pca_test.shape)

    X_train = make_sliding_windows(pca_train, window_size=window_size)
    X_val   = make_sliding_windows(pca_val, window_size=window_size)
    X_test  = make_sliding_windows(pca_test, window_size=window_size)

    print(f"Sliding window shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ---- Z-Normalization (one mean/std across all components) ----
    print(f"{pca_train.shape=}, {pca_val.shape=}, {pca_test.shape=}")
    mean = pca_train.mean()
    std = pca_train.std() + 1e-8  # add eps to avoid div by zero

    print(f"Used for normalization Mean: {mean}, Std: {std}")

    pca_train = (pca_train - mean) / std
    pca_val   = (pca_val - mean) / std
    pca_test  = (pca_test - mean) / std
    # -------------------------------------------------------------
    print("computing statistics of normalized data...")
    compute_statistics(pca_train)
    
    # Convert to tensors and save
    all_splits = {'train': [], 'val': [], 'test': []}

    all_splits['train'] = torch.tensor(X_train, dtype=torch.float32)
    all_splits['val'] = torch.tensor(X_val, dtype=torch.float32)
    all_splits['test'] = torch.tensor(X_test, dtype=torch.float32)

    print(f"{processed_dir=}")
    os.makedirs(processed_dir, exist_ok=True)

    for split_name, tensor in all_splits.items():
        save_path = os.path.join(processed_dir, f"{split_name}_{n_components}_{window_size}.pt")
        torch.save(tensor, save_path)   

    return all_splits


def one_hot_process_data_and_split(config, processed_dir='Datasets/one_hot_processed'):
    """
    Process raw data, split into train/val/test, apply sliding window, and save processed data.
    Replaces 20-D bout embeddings with a 13-D one-hot encoding from motor strategy labels.
    """
    file_path = config["data"]["raw_path"]                 # still used only to derive nonzero mask
    conditions_path = config["data"]["conditions_path"]
    window_size = config["data"]["sequence_length"]
    train_fraction = config["training"]["train_fraction"]
    val_fraction = config["training"]["val_fraction"]
    seed = config["seed"]

    # ---- Load embeddings solely to build the nonzero mask (as before) ----
    with h5py.File(file_path, 'r') as f:
        pca = np.array(f['pca_fish'])  # (n_fish, n_frames, n_components)
        nonzero_mask = np.any(pca != 0, axis=2)

    # ---- Load 13-class labels and convert to one-hot features ----
    labels_path = 'Datasets/JM_data/filtered_jmpool_kin.h5'
    with h5py.File(labels_path, 'r') as f:
        motor_strategies = np.array(f['bout_types'])  # expected shape (n_fish, n_frames), ints 0..12

    onehot_all = labels_to_onehot(motor_strategies, num_classes=13).astype(np.float32)  # (n_fish, n_frames, 13)
    n_fish = onehot_all.shape[0]

    conditions_idx = np.load(conditions_path)

    print(f"Total fish: {n_fish}")
    print(f"Original PCA shape: {pca.shape}, Nonzero mask shape: {nonzero_mask.shape}")
    print(f"Labels (motor_strategies) shape: {motor_strategies.shape} -> one-hot shape: {onehot_all.shape}")
    print(f"Conditions index shape: {conditions_idx.shape}")

    # ---- First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(n_fish),
        train_size=train_fraction,
        stratify=conditions_idx,
        random_state=seed
    )
    print(f"Train size: {len(train_idx)}, Val+Test size: {len(temp_idx)}")

    # ---- Second split: val vs test (from temp)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_fraction / (1 - train_fraction),
        stratify=conditions_idx[temp_idx],
        random_state=seed
    )
    print(f"Validation size: {len(val_idx)}, Test size: {len(test_idx)}")
    print(f"Validation: {val_idx}, Test: {test_idx}")

    with h5py.File(labels_path, 'r') as f:
        motor_strategies = np.array(f['bout_types'])  # (n_fish, n_frames)

    onehot_all, valid_mask = labels_to_onehot(motor_strategies, num_classes=13, return_valid_mask=True)

    # Combine masks: require both a nonzero PCA frame and a valid label
    combined_mask = nonzero_mask & valid_mask

    onehot_train = filter_nonzero(onehot_all[train_idx], combined_mask[train_idx])
    onehot_val   = filter_nonzero(onehot_all[val_idx],   combined_mask[val_idx])
    onehot_test  = filter_nonzero(onehot_all[test_idx],  combined_mask[test_idx])

    # # ---- Filter out zero frames using the same nonzero_mask derived from embeddings ----
    # onehot_train = filter_nonzero(onehot_all[train_idx], nonzero_mask[train_idx])
    # onehot_val   = filter_nonzero(onehot_all[val_idx],   nonzero_mask[val_idx])
    # onehot_test  = filter_nonzero(onehot_all[test_idx],  nonzero_mask[test_idx])

    print(f"Filtered one-hot shapes before sliding windowing: "
          f"train={onehot_train.shape}, val={onehot_val.shape}, test={onehot_test.shape}")
    print("------------------------------------------------")

    # ---- Sliding windows on one-hot features ----
    X_train = make_sliding_windows(onehot_train, window_size=window_size)
    X_val   = make_sliding_windows(onehot_val,   window_size=window_size)
    X_test  = make_sliding_windows(onehot_test,  window_size=window_size)

    print(f"Sliding window shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")



    # ---- Convert to tensors and save ----
    all_splits = {
        'train': torch.tensor(X_train, dtype=torch.float32),
        'val':   torch.tensor(X_val,   dtype=torch.float32),
        'test':  torch.tensor(X_test,  dtype=torch.float32),
    }

    os.makedirs(processed_dir, exist_ok=True)

    input_dim = 13
    for split_name, tensor in all_splits.items():
        save_path = os.path.join(processed_dir, f"{split_name}_{input_dim}_{window_size}.pt")
        torch.save(tensor, save_path)

    return all_splits






