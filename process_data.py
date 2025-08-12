import numpy as np
import torch
import h5py
import os
import matplotlib.pyplot as plt

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


def compute_statistics(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    min_val = torch.min(data, dim=0).values
    max_val = torch.max(data, dim=0).values

    print(f"Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")

    return mean, std, min_val, max_val


def process_data_fish_split():
    """
    Process data for fish split, currently a placeholder function.
    """
    raise NotImplementedError("This function is a placeholder and needs to be implemented.")