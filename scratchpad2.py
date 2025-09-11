import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File("Datasets/JM_data/filtered_jmpool_kin.h5") as f:
    motor_strategies_data = np.array(f["bout_types"])
    print(f"{motor_strategies_data.shape=}")  # Should be (463, 11651)
    print(list(f.keys()))
    orientations_data = np.array(f["orientation_smooth"])
    print(f"{orientations_data.shape=}")  # Should be (463, 11651)