import h5py
import numpy as np


with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
    data = f['MetaData']
    print(type(data))           # Check if it's a group or dataset
    print(list(data.keys()))    # If it's a group, list its contents
    print(list(f.keys()))    # If it's a group, list its contents
    head_speeds = f['speed_head'][:]

print(head_speeds)
print(head_speeds.shape)

