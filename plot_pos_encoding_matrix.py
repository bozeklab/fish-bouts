import math
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding_matrix(max_len: int, d_model: int) -> np.ndarray:
    """Return sinusoidal PE matrix of shape (max_len, d_model)."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(max_len, dtype=np.float32)[:, None]              # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32)
                      * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # even dims
    pe[:, 1::2] = np.cos(position * div_term)  # odd dims
    return pe

def generate_positional_encoding_heatmap(max_len: int, d_model: int,
                                         out_path: str | None = None,
                                         cmap=None):
    """
    Plot a heatmap of the sinusoidal positional encodings.

    Args:
        max_len: number of positions (y-axis).
        d_model: embedding dimension (x-axis).
        out_path: optional path to save the PNG.
        cmap: optional matplotlib colormap (e.g., 'coolwarm').
    """
    pe = positional_encoding_matrix(max_len, d_model)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(pe, aspect='auto', origin='upper', cmap=cmap) if cmap is not None \
         else plt.imshow(pe, aspect='auto', origin='upper')
    plt.title("Positional Encoding")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("")
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    plt.savefig("pos_enc112.png", dpi=200)
    return pe

# Example:
generate_positional_encoding_heatmap(10, 8, out_path="pos_enc.png", cmap='coolwarm')
