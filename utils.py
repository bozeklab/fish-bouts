import torch
import matplotlib.pyplot as plt


def apply_mask(x, mask_type='last', mask_ratio=0.3):
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
        x_masked[:, -num_mask:] = 0.0
        mask[:, -num_mask:] = True

    else:
        raise ValueError("mask_type must be 'random' or 'last'")

    return x_masked, mask


def compute_statistics(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    min_val = torch.min(data, dim=0).values
    max_val = torch.max(data, dim=0).values

    print(f"Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")

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
