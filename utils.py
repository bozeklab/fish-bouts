import torch
import matplotlib.pyplot as plt
import numpy as np

# def position_probs_from_onehot(x, weights, eps=1e-12):
#     """
#     x:           (B, K, C) one-hot tokens
#     weights:     (C,)      sampling weight/prob for each class (need not be normalized)
#     valid_mask:  (B, K)    optional bool mask of positions eligible for masking (True=ok)

#     returns:
#       probs: (B, K) valid probability distribution over positions (rows sum to 1)
#     """
#     B, K, C = x.shape
#     w = weights.view(1, 1, C)                   # (1,1,C)
#     scores = (x * w).sum(dim=-1)                # (B,K) weight of the class at each position

#     row_sums = scores.sum(dim=1, keepdim=True)  # (B,1)
#     probs = scores / torch.clamp(row_sums, min=eps)

#     return probs


# def sample_positions_to_mask(probs, num_masked_tokens):
#     """
#     probs: (B, K) rows sum to 1
#     returns:
#       idx: (B, num_masked_tokens) LongTensor with sampled indices per batch item
#     """
#     B, K = probs.shape
#     if num_masked_tokens > K:
#         raise ValueError("num_masked_tokens cannot exceed sequence length K when sampling without replacement.")
#     idx = torch.stack([
#         torch.multinomial(probs[b], num_masked_tokens, replacement=False)
#         for b in range(B)
#     ], dim=0)
#     return idx


def apply_mask(x, mask_type, mask_ratio, weights):
    """
    Apply zero masking to the input sequence.

    Args:
        x (Tensor): input tensor of shape (B, T, D)
        mask_type (str): 'random' or 'last'
        mask_ratio (float): fraction of positions to mask

    Returns:
        x_masked (Tensor): Masked input
        mask (BoolTensor): Mask indicating which positions were masked
    """
    with torch.no_grad():
        B, K, D = x.shape
        x_masked = x.clone()
        mask = torch.zeros(B, K, dtype=torch.bool)

        num_masked_tokens = max(1, min(K, int(round(mask_ratio * K))))

        if mask_type == 'random':
            ###################### OLD WAY (UNIFORM SAMPLING)
            # for i in range(B):
            #     mask_idx = torch.randperm(K)[:num_masked_tokens]
            #     print(f"{mask_idx.shape=}")
            #     print(f"{mask_idx=}")
            #     x_masked[i, mask_idx] = 0.0
            #     mask[i, mask_idx] = True
            ####################### NEW WAY (WEIGHTED SAMPLING)
            if weights is None:
                # Uniform sampling
                probs = torch.ones(B, K, device=x.device) / K
            else:
                # # w = weights.view(1, 1, D) 
                # # scores = (x_masked * w).sum(dim=-1)
                # print(f"{scores.shape=}")
                # print(f"{scores=}")
                # row_sums = scores.sum(dim=1, keepdim=True)  # (B,1)
                # print(f"{row_sums.shape=}")
                # probs = scores / torch.clamp(row_sums, min=1e-12)
                device = x.device
                w = torch.as_tensor(weights, device=device, dtype=torch.float32)
                w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)

                scores = (x.to(torch.float32) * w.view(1,1,-1)).sum(dim=-1)  # (B,K) ≥ 0

                eps = 1e-12
                logits = torch.log(scores.clamp_min(eps))   # log turns linear weights into logits
                probs  = torch.softmax(logits, dim=1)       # (B,K) strictly > 0, rows sum to 1

            masked_indices = torch.multinomial(probs, num_samples=num_masked_tokens, replacement=False) 
            
            row_idx = torch.arange(B, device=masked_indices.device).unsqueeze(1)   # (B,1)
            x_masked[row_idx, masked_indices] = 0.0
            mask[row_idx, masked_indices] = True

        elif mask_type == 'last':
            x_masked[:, -num_masked_tokens:] = 0.0
            mask[:, -num_masked_tokens:] = True
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
