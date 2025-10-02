import numpy as np
import torch
import h5py
import os
from sklearn.model_selection import train_test_split

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


def labels_to_onehot(labels, num_classes):
    """
    """
    print(f"{labels=}")
    labels = np.asarray(labels, dtype=np.int64)

    print(f"{labels.dtype=}")
    # Coerce to integers robustly
    # if lab.dtype.kind in ('U', 'S', 'O'):  # strings/bytes/object
    #     lab = lab.astype(np.int64)
    # elif lab.dtype.kind == 'f':            # floats
    #     if not np.all(np.isfinite(lab)):
    #         raise ValueError("Labels contain non-finite values (NaN/Inf).")
    #     lab = np.rint(lab).astype(np.int64)
    # elif lab.dtype.kind in ('i', 'u'):     # already integer/unsigned
    #     lab = lab.astype(np.int64)
    # else:
    #     raise TypeError(f"Unsupported labels dtype: {lab.dtype}")

    valid_mask = (labels >= 0) & (labels < num_classes)
    eye = np.eye(num_classes, dtype=np.float32)
    onehot = np.zeros(labels.shape + (num_classes,), dtype=np.float32)
    onehot[valid_mask] = eye[labels[valid_mask]]

    return onehot


def process_data_and_split(config, one_hot_encode):
    """
    Process raw data, split into train/val/test, apply sliding window, and save processed data.
    If one_hot_encode==True, replace 20-dim bout embeddings with 13-dim one-hot encodings from motor strategy labels.
    """
    file_path = config["data"]["raw_path"]
    conditions_path = config["data"]["conditions_path"]
    window_size = config["data"]["sequence_length"]
    train_fraction = config["training"]["train_fraction"]
    val_fraction = config["training"]["val_fraction"]
    processed_dir = config["data"]["processed_dir"]
    seed = config["seed"]

    with h5py.File(file_path, 'r') as f:
        pca = np.array(f['pca_fish'])  # (n_fish, n_frames, n_components)
        n_fish, n_frames, n_components = pca.shape
        # different fish have different n_frames, rest is zero-padded
        nonzero_mask = np.any(pca != 0, axis=2) # non-zero bouts

    with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
        class_labels = np.array(f['bout_types']) # 1-13, 15 for padding
        class_labels -= 1  # convert to 0-based  (0-12, 14 for padding)

    # filter out the zero_padding
    class_labels_nonzero = class_labels[nonzero_mask]

    # check the number of distinct classes
    unique_labels = np.unique(class_labels_nonzero)
    n_classes = len(unique_labels) # should be 13
    print(f"{n_classes=}")

    if one_hot_encode:
        fish_embeddings = labels_to_onehot(class_labels, num_classes=n_classes)#.astype(np.float32)  # (_fish, n_frames, n_classes)
    else:
        fish_embeddings = pca

    # load the data on conditions that different fish were in
    # train-val-test split should respect that distribution
    conditions = np.load(conditions_path)

    # Step 1: split the dataset into train-val-test, respecting the distribution of conditions
    # first split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(n_fish),
        train_size=train_fraction,
        stratify=conditions, # respect the distribution of different conditions
        random_state=seed
    )

    # second split: val vs test (from temp)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_fraction / (1 - train_fraction),
        stratify=conditions[temp_idx], # respect the distribution of different conditions
        random_state=seed
    )

    # Step 2: for each data split, filter the non-zero bouts
    nonzero_embeddings_train = filter_nonzero(fish_embeddings[train_idx], nonzero_mask[train_idx])
    nonzero_embeddings_val   = filter_nonzero(fish_embeddings[val_idx],   nonzero_mask[val_idx])
    nonzero_embeddings_test  = filter_nonzero(fish_embeddings[test_idx],  nonzero_mask[test_idx])

    nonzero_labels_train = filter_nonzero(class_labels[train_idx], nonzero_mask[train_idx])
    nonzero_labels_val   = filter_nonzero(class_labels[val_idx],   nonzero_mask[val_idx])
    nonzero_labels_test  = filter_nonzero(class_labels[test_idx],  nonzero_mask[test_idx])

    # Step 3: Sliding windows on the non-zero embeddings
    X_train = make_sliding_windows(nonzero_embeddings_train, window_size=window_size)
    X_val   = make_sliding_windows(nonzero_embeddings_val,   window_size=window_size)
    X_test  = make_sliding_windows(nonzero_embeddings_test,  window_size=window_size)

    Y_train = make_sliding_windows(nonzero_labels_train[:, None], window_size=window_size)
    Y_val   = make_sliding_windows(nonzero_labels_val[:, None],   window_size=window_size)
    Y_test  = make_sliding_windows(nonzero_labels_test[:, None],  window_size=window_size)

    print(f"Sliding window shapes: \ntrain={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Sliding window shapes: \ntrain={Y_train.shape}, val={Y_val.shape}, test={Y_test.shape}")

    def bundle(x, y):
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }

    splits = {
        "train": bundle(X_train, Y_train),
        "val":   bundle(X_val,   Y_val),
        "test":  bundle(X_test,  Y_test),
    }

    os.makedirs(processed_dir, exist_ok=True)

    input_dim = X_train.shape[-1]

    for split_name, tensor in splits.items():
        save_path = os.path.join(processed_dir, f"{split_name}_{input_dim}_{window_size}.pt")
        torch.save(tensor, save_path)

    return splits






