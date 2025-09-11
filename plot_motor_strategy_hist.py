import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --- Bout types (motor strategies: J-turn etc.) histogram ---

# Load motor strategies data from 
# filtered_jmpool_kin.h5 --> bout_types
with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
    motor_strategies_data = np.array(f['bout_types']) 
    
print(f"{motor_strategies_data.shape=}")
print(f"{motor_strategies_data=}")
# values 1-13 correspond to motor strategies
# value 15 is for padding

motor_strategies_names = np.load('Datasets/JM_data/classnames_jm.npy', allow_pickle=True)
print(f"{motor_strategies_names=}") 


# Define motor strategies high-level categories
idx_to_cat = {0: "Displacement", 1: "Reorienting", 2: "Other"}



motor_strategies_data_filtered = motor_strategies_data[motor_strategies_data != 15]  # remove padding
print(f"{motor_strategies_data_filtered.shape=}")


# Categories
categories_dict = {0: "Displacement", 1: "Reorienting", 2: "Other"}
category_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

# Motor strategies with categories
ms_categories = np.array([
    ['Short_CS', 2],
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
    ['O-bend', 0]
], dtype=object)

ms_categories_sorted = ms_categories[np.argsort(ms_categories[:, 1].astype(int))]

motor_strategies_names = ms_categories_sorted[:, 0]

counts, _ = np.histogram(
    motor_strategies_data_filtered,
    bins=np.arange(1, len(ms_categories) + 2) - 0.5
)
percentages = counts / len(motor_strategies_data_filtered)

percentage_dict = {name: percentages[i] for i, name in enumerate(ms_categories[:, 0])}
percentages_sorted = [percentage_dict[name] for name in motor_strategies_names]

# assign colors by category
bar_colors = [category_colors[int(cat)] for cat in ms_categories_sorted[:, 1]]

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.bar(
    motor_strategies_names,
    percentages_sorted,
    color=bar_colors,
    width=0.8
)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Motor strategy (bout type)")
plt.ylabel("Percentage")
plt.title("Motor strategies distribution")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.grid(axis="y", linestyle="--", alpha=0.7)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=category_colors[c], label=categories_dict[c])
    for c in categories_dict
]
plt.legend(handles=handles, title="Categories")

plt.tight_layout()
plt.savefig("motor_strategies_histogram.png")
plt.close()
