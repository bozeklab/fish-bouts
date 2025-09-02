import torch
import matplotlib.pyplot as plt
import numpy as np

def apply_mask(x, mask_type='last', mask_ratio=0.3, if_print=False):
    """
    Apply masking to the input sequence.

    Args:
        x (Tensor): Input tensor of shape (B, T, D)
        mask_type (str): 'random' or 'last_k'
        mask_ratio (float): Fraction of positions to mask

    Returns:
        x_masked (Tensor): Masked input
        mask (BoolTensor): Mask indicating which positions were masked
    """
    B, T, D = x.shape
    x_masked = x.clone()
    mask = torch.zeros(B, T, dtype=torch.bool)

    num_mask = max(1, int(mask_ratio * T))

    if mask_type == 'random':
        for i in range(B):
            mask_idx = torch.randperm(T)[:num_mask]
            x_masked[i, mask_idx] = 0.0
            mask[i, mask_idx] = True

    elif mask_type == 'last':
        # if if_print:
        #     print(f"{x_masked[:, -num_mask:]=}")
        x_masked[:, -num_mask:] = 0.0
        mask[:, -num_mask:] = True
    else:
        raise ValueError("mask_type must be 'random' or 'last'")

    return x_masked, mask


def compute_statistics(data):
    print("Computing statistics...")
    
    print("data shape", data.shape)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, positions=np.arange(1, 21), flierprops={'marker': '.', 'markersize': 1, 'markerfacecolor': 'gray'})

    # Overlay mean, min, max, and ±1 std
    """plt.plot(np.arange(1, 21), mean, "o-", label="Mean", color="red")
    plt.plot(np.arange(1, 21), min_val, "x-", label="Min", color="blue")
    plt.plot(np.arange(1, 21), max_val, "x-", label="Max", color="green")
    plt.plot(np.arange(1, 21), mean - std, "--", label="Mean - Std", color="orange")
    plt.plot(np.arange(1, 21), mean + std, "--", label="Mean + Std", color="orange")"""

    #plt.xlabel("Component")
    #plt.ylabel("Value")
    plt.xticks(np.arange(1, 21))
    plt.legend()
    plt.savefig("statistics_plot.png")

    return mean, std, min_val, max_val


def plot_loss_curve(history):
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
