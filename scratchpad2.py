import numpy as np
import h5py
data = np.load('Datasets/JM_data/cg2_labels.npy', allow_pickle=True)
print(type(data))        # Check the type of the loaded object
print(data.shape)              # Print the contents or inspect further
print(data[:100])  # Print the data to see its structure and contents
print(np.sum(data))  # Example operation: sum of all elements

data = np.load('Datasets/JM_data/cg4_labels.npy', allow_pickle=True)
print(type(data))        # Check the type of the loaded object
print(data.shape)              # Print the contents or inspect further
print(data[:100])  # Print the data to see its structure and contents
print(np.sum(data))  # Example operation: sum of all elements

data = np.load('Datasets/JM_data/cg7_labels.npy', allow_pickle=True)
print(type(data))        # Check the type of the loaded object
print(data.shape)              # Print the contents or inspect further
print(data[:100])  # Print the data to see its structure and contents
print(np.sum(data))  # Example operation: sum of all elements


data = np.load('Datasets/JM_data/classnames_jm.npy', allow_pickle=True)
print(type(data))        # Check the type of the loaded object
print(data.shape)              # Print the contents or inspect further
print(data[:100])  # Print the data to see its structure and contents



with h5py.File('Datasets/JM_data/filtered_jmpool_kin.h5', 'r') as f:
    data = f['MetaData']
    print(type(data))           # Check if it's a group or dataset
    print(list(data.keys()))    # If it's a group, list its contents
    print(f"{data['lengths_data'].shape=}")
    print(f"{data['errmask'].shape=}")
    print(f"{data['t0_bout'][:]=}")
    print(f"{data['frameRate'][:]=}")

print("---------------------------------------------")
with h5py.File('Datasets/JM_data/simlabels_fish_K5_N1200_tau3_cg2.h5', 'r') as f:
    data = f['MetaData']
    print(type(data))           # Check if it's a group or dataset
    print(list(data.keys()))    # If it's a group, list its contents
    print(f"{data['lengths_sims']=}")
    print(f"{data['n_seeds'].shape=}")
    print(f"{data['simfishes'][:]=}")


print("---------------------------------------------")
with h5py.File('Datasets/JM_data/simlabels_fish_K5_N1200_tau3_cg4.h5', 'r') as f:
    data = f['MetaData']
    print(type(data))           # Check if it's a group or dataset
    print(list(data.keys()))    # If it's a group, list its contents
    print(f"{data['lengths_sims']=}")
    print(f"{data['n_seeds'].shape=}")
    print(f"{data['simfishes'][:]=}")

print("---------------------------------------------")
with h5py.File('Datasets/JM_data/simlabels_fish_K5_N1200_tau3_cg7.h5', 'r') as f:
    data = f['MetaData']
    print(type(data))           # Check if it's a group or dataset
    print(list(data.keys()))    # If it's a group, list its contents
    print(f"{data['lengths_sims']=}")
    print(f"{data['n_seeds'].shape=}")
    print(f"{data['simfishes'][:]=}")

print("---------------------------------------------")
print("spectral_split_g7.npy")
data = np.load('Datasets/JM_data/spectral_split_g7.npy', allow_pickle=True)
print(data.shape)              # Print the contents or inspect further
print(data[:])  # Print the data to see its structure and contents

print("---------------------------------------------")
print("posterior.npy")
data = np.load('Datasets/JM_data/posterior.npy', allow_pickle=True)
print(data.shape)              # Print the contents or inspect further
print(data[:])  # Print the data to see its structure and contents
print(np.sum(data, axis=1))  # Example operation: sum of all elements

print("---------------------------------------------")
print("spectral_split_g7.npy")
data = np.load('Datasets/JM_data/spectral_split_g7.npy', allow_pickle=True)
print(data.shape)              # Print the contents or inspect further
print(data[:])  # Print the data to see its structure and contents