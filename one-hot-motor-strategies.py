import numpy as np
import h5py
import os
import matplotlib
matplotlib.use("Agg")  # headless backend for clusters
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


# conditions_idx = np.load('sensory_contexts_data.npy')
# print(f"{conditions_idx.shape=}")

# LOAD THE J-TURN ETC.
with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
    motor_strategies = np.array(f['bout_types']) 
    print(f"{motor_strategies.shape=}")

print(f"{motor_strategies=}")




# ---------- helpers ----------
def _load_classnames(path="classnames_jmnpy"):
    """
    Tries to load class names from:
      - a .npy file (array of strings, or a dict mapping value->name), or
      - a .npz file (first array), or
      - a .txt file (one name per line)
    You can pass 'classnames_jmnpy' with or without extension.
    Returns either a list/array of names or a dict {int_value: name}.
    """
    candidates = [path]
    # Try some common extensions if none given
    if "." not in os.path.basename(path):
        candidates += [path + ".npy", path + ".npz", path + ".txt"]

    for p in candidates:
        if not os.path.exists(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".npy":
                arr = np.load(p, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.ndim == 0:
                    obj = arr.item()
                    if isinstance(obj, dict):
                        return {int(k): str(v) for k, v in obj.items()}
                    arr = np.asarray(obj)
                return arr
            elif ext == ".npz":
                npz = np.load(p, allow_pickle=True)
                key = list(npz.keys())[0]
                arr = npz[key]
                return arr
            elif ext == ".txt":
                with open(p, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
            continue
    raise FileNotFoundError(
        f"Could not load class names from '{path}'. "
        f"Tried: {', '.join(candidates)}"
    )

def _build_value_to_name(unique_vals, classnames):
    """
    Make {value -> name} mapping.
    - If classnames is a dict, use it directly (by int(value)).
    - If classnames is a list/array of strings:
        * If len == len(unique_vals): map in the order of sorted unique_vals.
        * Else if len > max(unique_val_int): map by index (int(value)).
        * Else fall back to 'Class {int(value)}'.
    """
    unique_vals_sorted = np.sort(unique_vals)
    v2n = {}
    if isinstance(classnames, dict):
        for v in unique_vals_sorted:
            v_int = int(round(float(v)))
            v2n[v] = classnames.get(v_int, f"Class {v_int}")
        return v2n

    names = list(np.asarray(classnames).astype(str))
    if len(names) == len(unique_vals_sorted):
        for v, name in zip(unique_vals_sorted, names):
            v2n[v] = name
    elif len(names) > int(np.max(unique_vals_sorted)):
        for v in unique_vals_sorted:
            v2n[v] = names[int(round(float(v)))]
    else:
        for v in unique_vals_sorted:
            v2n[v] = f"Class {int(round(float(v)))}"
    return v2n


def save_dotplot_per_row(
    motor_strategies,
    classnames_path="classnames_jmnpy",
    out_dir="motor_strategy_plots",
    prefix="fish",
    fmt="png",
    figsize=(12, 3.5),
    downsample=None,
    dpi=150,
    equal_value=15.0,
    equal_tol=1e-8,
    markersize=6,
    ytick_fontsize=8,
    line_width=0.6,          # connecting line width
    line_color="black",      # connecting line color
    show_hlines=True,        # NEW: draw horizontal dotted lines
    hline_style=":",         # NEW: dotted
    hline_width=0.6,         # NEW: thin guide lines
    hline_color="0.6",       # NEW: neutral gray
    hline_alpha=0.9          # NEW: slightly transparent
):
    """
    Saves one dot-plot per file.
    - Filters out points equal to `equal_value`.
    - Colors encode class value.
    - Y-axis tick labels show class NAMES (serves as legend).
    - Connects visible dots with thin black lines (no bridges over gaps).
    - Adds horizontal dotted guide lines at each class level.
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = np.asarray(motor_strategies)

    # Optional speed-up
    if downsample and downsample > 1:
        arr = arr[:, ::downsample]

    # Global mapping ensures consistent colors and y-ticks across files
    mask_keep = ~np.isclose(arr, equal_value, atol=equal_tol)
    kept_vals = arr[mask_keep]
    unique_vals = np.unique(kept_vals) if kept_vals.size else np.array([], dtype=float)

    classnames = _load_classnames(classnames_path)
    v2n = _build_value_to_name(unique_vals, classnames)
    values_sorted = np.sort(unique_vals)
    k = len(values_sorted)
    cmap = _prepare_palette(max(k, 1))
    val_to_idx = {v: i for i, v in enumerate(values_sorted)}

    # Precompute y-ticks/labels used on every plot
    if k > 0:
        ytick_positions = values_sorted
        ytick_labels = [v2n[v] for v in values_sorted]
        ymin, ymax = float(values_sorted.min()) - 0.5, float(values_sorted.max()) + 0.5
    else:
        ytick_positions, ytick_labels = [], []
        ymin = ymax = None

    for i, y in enumerate(arr):
        fig, ax = plt.subplots(figsize=figsize)
        n_samples = y.size

        keep = ~np.isclose(y, equal_value, atol=equal_tol)
        if np.any(keep):
            x = np.nonzero(keep)[0]
            yk = y[keep].astype(float)
            cidx = np.array([val_to_idx.get(v, 0) for v in yk], dtype=int)

            # Connect consecutive kept samples only (no bridging gaps)
            if x.size > 1:
                breaks = np.where(np.diff(x) > 1)[0] + 1
                starts = np.r_[0, breaks]
                ends = np.r_[breaks, x.size]
                for s, e in zip(starts, ends):
                    if e - s >= 2:
                        ax.plot(
                            x[s:e], yk[s:e],
                            linewidth=line_width,
                            color=line_color,
                            zorder=1
                        )

            # Dots on top of the lines
            ax.scatter(
                x, yk,
                c=cidx, cmap=cmap, vmin=0, vmax=max(k-1, 0),
                s=markersize, edgecolors="none", rasterized=True, zorder=2
            )
            ax.set_xlabel("Bout index")
            xmin, xmax = ax.get_xlim()
        else:
            ax.text(0.5, 0.5, "all values == 15", ha="center", va="center")
            ax.set_xticks([])
            # Make guides span the full row width even if no points are shown
            xmin, xmax = -0.5, max(n_samples - 0.5, 0.5)
            ax.set_xlim(xmin, xmax)

        # Y-axis as the "legend": show class names at their numeric values
        ax.set_ylabel("Motor strategy")
        if k > 0:
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels, fontsize=ytick_fontsize)
            ax.set_ylim(ymin, ymax)

            # --- NEW: horizontal dotted guide lines at each class level ---
            if show_hlines:
                ax.hlines(
                    ytick_positions, xmin, xmax,
                    linestyles=hline_style,
                    linewidths=hline_width,
                    colors=hline_color,
                    alpha=hline_alpha,
                    zorder=0
                )
                # Preserve computed x-limits
                ax.set_xlim(xmin, xmax)
        else:
            ax.set_yticks([])

        ax.set_title(f"Fish {i}")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{prefix}_{i:03d}.{fmt}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _prepare_palette(k):
    """
    Return a ListedColormap with at least k distinct colors.
    Uses 'tab20' which has 20 discrete colors; will cycle if k > 20.
    """
    base = plt.get_cmap("tab20").colors
    colors = [base[i % len(base)] for i in range(k)]
    return ListedColormap(colors)



save_dotplot_per_row(
    motor_strategies,
    classnames_path="Datasets/JM_data/classnames_jm.npy",
    downsample=5,
    show_hlines=True,       # turn on
    hline_style=":",        # dotted
    hline_width=0.6,
    hline_color="0.6",
    hline_alpha=0.9
)


print("Plotting finished!")






