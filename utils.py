import torch
import matplotlib.pyplot as plt
import numpy as np


def apply_mask(x, y, mask_type, mask_ratio, weights, one_hot_encoded=False):
    """
    Apply zero masking to the input sequence.

    Args:
        x (Tensor): input tensor of shape (B, T, D)
        mask_type (str): 'random' or 'last'
        mask_ratio (float): fraction of positions to mask

    Returns:
        x_masked (Tensor): masked input 
        mask (BoolTensor): mask indicating which positions were masked
    """
    device = x.device
    with torch.no_grad():
        B, K, D = x.shape
        x_masked = x.clone()
        mask = torch.zeros(B, K, dtype=torch.bool)

        print(f"{x=}")
        print(f"{x.shape=}")
        # Ensure the number of masked tokens is between 1 and K
        num_masked_tokens = max(1, min(K, int(round(mask_ratio * K)))) 

        if mask_type == 'random':
            if weights is None:
                # Uniform sampling
                probs = torch.ones(B, K, device=x.device) / K
            else:
                # Non-uniform, weighted sampling taking into account imbalance of classes
                if one_hot_encoded:
                    # Weighted sampling for one-hot encoded inputs
                    w = torch.as_tensor(weights, device=device, dtype=torch.float32)
                    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
                    scores = (x.to(torch.float32) * w.view(1,1,-1)).sum(dim=-1)
                    probs = scores / scores.sum(dim=-1, keepdim=True)
                else: 
                    # Weighted sampling for pca embeddings
                    frequencies = torch.bincount(y)

                # w = torch.as_tensor(weights, device=device, dtype=torch.float32)
                # w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
                # scores = (x.to(torch.float32) * w.view(1,1,-1)).sum(dim=-1)
                # probs = scores / scores.sum(dim=-1, keepdim=True)
                # --------
                

            print(f"{probs.device=}")
            

            masked_indices = torch.multinomial(probs, num_samples=num_masked_tokens, replacement=False)
            print(f"{masked_indices.shape=}")
            
            row_idx = torch.arange(B, device=masked_indices.device).unsqueeze(1)

            x_masked[row_idx, masked_indices] = 0.0
            mask[row_idx, masked_indices] = True

        elif mask_type == 'last':
            # Mask the last num_masked_tokens tokens
            x_masked[:, -num_masked_tokens:] = 0.0
            mask[:, -num_masked_tokens:] = True
        else:
            raise ValueError("mask_type must be 'random' or 'last'")
        
        # --- Global count of masked classes across the whole batch ---
        # Recover class ids from one-hot x BEFORE masking
        class_ids = torch.argmax(x, dim=-1)  # (B, K), dtype long
        masked_labels = class_ids[mask]      # 1D tensor of masked class ids

        masked_classes_array = torch.bincount(masked_labels, minlength=D)  # (D,)

    return x_masked, mask, masked_classes_array


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


