# fish-bouts

The structure of the raw data (available at https://doi.org/10.5281/zenodo.13605471) is as follows:

```
├── Datasets
│ ├── JM_data
│ │ ├── cg2_labels.npy
│ │ ├── cg4_labels.npy
│ │ ├── cg7_labels.npy
│ │ ├── classnames_jm.npy
│ │ ├── eigfs_n1200.npy
│ │ ├── Entropy_seeds_delays_clusters.h5
│ │ ├── filtered_jmpool_kin.h5
│ │ ├── kmeans_labels_K5_N1200_s8684.h5
│ │ ├── P_ensemble_ex8_N1200_s8684.npy
│ │ ├── pool_ex8_PCs.h5
│ │ ├── posterior.npy
│ │ ├── simlabels_fish_K5_N1200_tau3_cg2.h5
│ │ ├── simlabels_fish_K5_N1200_tau3_cg4.h5
│ │ ├── simlabels_fish_K5_N1200_tau3_cg7.h5
│ │ ├── spectral_split_g7.npy
│ │ └── zebrafish_ms_sims
│ └── Ph_Data
│ ├── Entropy_seeds_delays_clusters.h5
│ ├── filtered_phdata9_condition_3.h5
│ └── kmeans_labels_K7_N1100.h5
```

There are two separate datasets: JM_data and Ph_Data.

| Feature              | JM_data (1)                 | Ph_Data (8)               |
|----------------------|-----------------------------|---------------------------|
| Number of fish       | 463                         | 218                       |
| Species              | Tubingen zebrafish          | AB zebrafish              |
| Temporal resolution  | 700 Hz                      | 160 Hz                    |
| Pixel size           | 58 µm / 27 µm (depending on arena's size)               | 70 µm                     |
| Tail points          | 8                           | 8                         |
| Max frames (duration) | 175 frames (250 ms)        | 40 frames (250 ms)        |
| Environment conditions | 14 sensory contexts      | acidic/non-acidic      |
| Tracking             | custom software        | Zebrazoom       |

