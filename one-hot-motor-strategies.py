import numpy as np
import h5py
import os
import matplotlib
matplotlib.use("Agg")  # headless backend for clusters
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# conditions_idx = np.load('sensory_contexts_data.npy')
# print(f"{conditions_idx.shape=}")

# LOAD THE J-TURN ETC.
with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
    motor_strategies = np.array(f['bout_types']) 
    print(f"{motor_strategies.shape=}")

print(f"{motor_strategies=}")

ms_categories = np.array([['Short_CS', 2],
       ['AS', 2],
       ['Slow1', 2],
       ['Slow2', 2],
       ['J_turn', 1],
       ['HAT', 1],
       ['RT', 1],
       ['BS', 0],
       ['Long_CS', 0],
       ['LLC', 0],
       ['SLC', 0],
       ['SAT', 0],
       ['O-bend', 0]], dtype=object)

print(f"{ms_categories=}")

categories_dict = {0: "Displacement bout", 1: "Reorienting bout", 2: "Forward-Displacing bout"}

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


def _normalize_group_lookup(group_index_lookup):
    """
    Accepts either:
      - dict {strategy_name -> 0|1|2}, or
      - 2D array-like [[strategy_name, 0|1|2], ...]
    Returns dict {str(name): int(group)}.
    """
    if group_index_lookup is None:
        return None
    if isinstance(group_index_lookup, dict):
        return {str(k): int(v) for k, v in group_index_lookup.items()}
    arr = np.asarray(group_index_lookup, dtype=object)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return {str(arr[i, 0]): int(arr[i, 1]) for i in range(arr.shape[0])}
    raise ValueError("group_index_lookup must be a dict or 2D array of [name, group].")


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
    line_width=0.6,
    line_color="black",
    show_hlines=True,
    hline_style=":",
    hline_width=0.6,
    hline_color="0.6",
    hline_alpha=0.9,
    # [NEW] three-color grouping by indices 0/1/2
    three_color_mode=True,           # [NEW]
    group_index_lookup=None,         # [NEW] pass [['Short_CS',0], ...] or dict
    three_cmap=None                  # [NEW] optional custom 3-color ListedColormap
):
    """
    Saves one dot-plot per file.

    When three_color_mode=True:  # [NEW]
      - Points are colored by a *group index* in {0,1,2}, not by raw class value.  # [NEW]
      - The group index is looked up by strategy name using `group_index_lookup`.   # [NEW]
      - Y-axis tick labels remain per class value (from classnames_path).           # [NEW]
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = np.asarray(motor_strategies)

    # if downsample and downsample > 1:
    #     arr = arr[:, ::downsample]
    print(f"{arr=}")
    print(f"{equal_value=}")
    mask_keep = ~np.isclose(arr, equal_value, atol=equal_tol)
    kept_vals = arr[mask_keep]
    unique_vals = np.unique(kept_vals) if kept_vals.size else np.array([], dtype=float)

    #classnames = _load_classnames(classnames_path)
    classnames = ms_categories[:, 0]
    # = np.array([['Short_CS', 0],
    #    ['AS', 0],
    #    ['Slow1', 0],
    #    ['Slow2', 0],
    #    ['J_turn', 1],
    #    ['HAT', 1],
    #    ['RT', 1],
    #    ['BS', 2],
    #    ['Long_CS', 2],
    #    ['LLC', 2],
    #    ['SLC', 2],
    #    ['SAT', 2],
    #    ['O-bend', 2]], dtype=object)

    v2n = _build_value_to_name(unique_vals, classnames)
    values_sorted = np.sort(unique_vals)
    k = len(values_sorted)

    # --- color handling ---
    default_cmap = _prepare_palette(max(k, 1))  # original palette

    # [NEW] fixed 3-color palette (default to first 3 colors from tab10)
    if three_cmap is None:
        three_cmap = ListedColormap(plt.get_cmap("tab10").colors[:3])

    # [NEW] normalize incoming lookup to dict {name -> 0/1/2}
    group_lookup = _normalize_group_lookup(group_index_lookup) if three_color_mode else None

    # Precompute y-ticks/labels
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

            # [CHANGED] choose color indices:
            if three_color_mode and group_lookup is not None:
                # [NEW] map each value -> class name -> group 0/1/2
                names = [v2n.get(v, str(int(round(v)))) for v in yk]        # [NEW]
                cidx = np.array([int(group_lookup.get(name, 0)) for name in names], dtype=int)  # [NEW]
                cmap_for_scatter = three_cmap                                                # [NEW]
                vmin, vmax = 0, 2                                                             # [NEW]
            else:
                # original per-class coloring
                val_to_idx = {v: i for i, v in enumerate(values_sorted)}
                cidx = np.array([val_to_idx.get(v, 0) for v in yk], dtype=int)
                cmap_for_scatter = default_cmap
                vmin, vmax = 0, max(k - 1, 0)

            # Connect consecutive kept samples only
            if x.size > 1:
                breaks = np.where(np.diff(x) > 1)[0] + 1
                starts = np.r_[0, breaks]
                ends = np.r_[breaks, x.size]
                for s, e in zip(starts, ends):
                    if e - s >= 2:
                        ax.plot(x[s:e], yk[s:e], linewidth=line_width, color=line_color, zorder=1)

            ax.scatter(
                x, yk,
                c=cidx, cmap=cmap_for_scatter, vmin=vmin, vmax=vmax,
                s=markersize, edgecolors="none", rasterized=True, zorder=2
            )
            if three_color_mode and group_lookup is not None:
                from matplotlib.lines import Line2D

                present_groups = np.unique(cidx)  # which of {0,1,2} appear in this plot
                handles = [
                    Line2D([0], [0],
                           marker='o', linestyle='none',
                           markersize=markersize * 0.8,
                           markerfacecolor=three_cmap(int(g)),
                           markeredgecolor='none',
                           label=categories_dict.get(int(g), f"Group {int(g)}"))
                    for g in present_groups
                ]
                if handles:
                    ax.legend(
                        handles=handles,
                        #title="Bout Categories",
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),   # outside, right side
                        frameon=False,
                        fontsize=7,                   # smaller labels
                        title_fontsize=8              # smaller title
                    )
            ax.set_xlabel("Bout index")
            xmin, xmax = ax.get_xlim()
        else:
            ax.text(0.5, 0.5, "all values == 15", ha="center", va="center")
            ax.set_xticks([])
            xmin, xmax = -0.5, max(n_samples - 0.5, 0.5)
            ax.set_xlim(xmin, xmax)

        ax.set_ylabel("Motor strategy")
        if k > 0:
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels, fontsize=ytick_fontsize)
            ax.set_ylim(ymin, ymax)

            if three_color_mode and group_lookup is not None:
                for label in ax.get_yticklabels():
                    name = label.get_text()
                    g = group_lookup.get(name, None)
                    if g is not None:
                        label.set_color(three_cmap(int(g)))

            if show_hlines:
                ax.hlines(
                    ytick_positions, xmin, xmax,
                    linestyles=hline_style, linewidths=hline_width,
                    colors=hline_color, alpha=hline_alpha, zorder=0
                )
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
    hline_alpha=0.9,
    group_index_lookup=ms_categories,   # [NEW] provide grouping
)


print("Plotting finished!")






