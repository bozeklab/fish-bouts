# Dataset Description

The folder structure of the data (available at https://doi.org/10.5281/zenodo.13605471) is as follows:

```
├── Datasets
│   ├── JM_data
│   └── Ph_Data
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
| Citation   | JC Marques, S Lackner, R Félix, MB Orger, Structure of the zebrafish locomotor repertoire revealed with unsupervised behavioral clustering. Curr. Biol. 28, 181–195 (2018).   |  G Reddy, et al., A lexical approach for identifying behavioural action sequences. PLoS computational biology 18, e1009672 (2022).      |

We focus on the JM_data. 
```
├── JM_data
│   ├── cg2_labels.npy
│   ├── cg4_labels.npy
│   ├── cg7_labels.npy
│   ├── classnames_jm.npy
│   ├── eigfs_n1200.npy
│   ├── Entropy_seeds_delays_clusters.h5
│   ├── filtered_jmpool_kin.h5
│   ├── kmeans_labels_K5_N1200_s8684.h5
│   ├── P_ensemble_ex8_N1200_s8684.npy
│   ├── pool_ex8_PCs.h5
│   ├── posterior.npy
│   ├── simlabels_fish_K5_N1200_tau3_cg2.h5
│   ├── simlabels_fish_K5_N1200_tau3_cg4.h5
│   ├── simlabels_fish_K5_N1200_tau3_cg7.h5
│   ├── spectral_split_g7.npy
│   └── zebrafish_ms_sims
```

The raw data (i.e. the 8 tail angles x 175 frames sequences) is very large and thus not included. 

- ```cg2_labels.npy```
- ```cg4_labels.npy```
- ```cg7_labels.npy```

contain a classification of each of the 1200 microstates into one of 2/4/7 metastable states (motor strategies) respectively.

For q=2: low (“cruising”) and high (“wandering”) rate of reorientation.

For q=4: slow and fast variations of cruising and wandering.

For q=7: left/right variants of slow and fast wandering, and of fast cruising:
1. left slow wandering
2. right slow wandering
3. left fast wandering
4. right fast wandering
5. left fast cruising
6. right fast cruising
7. slow cruising

- ```classnames_jm.npy``` contains names of the 13 different motor strategies (J_turn, Approach Swim etc.).
- ```pool_ex8_PCs.h5``` contains
  - 'pca_fish': the first 50 PCA components of the fish bouts (out of which only first 20 are used).
  - 'var_exp': the amount of variance explained by the PCA components.
  - 'eigvecs': 
  - 'data_means': 
  - 'cov': 
  - 'max_shuffs'
  - 'seeds': values of 10 random seeds used
- ```spectral_split_g7.npy``` contains classification of each of 463 fish into one of 7 classes.
- ```posterior.npy``` contains for each of 463 fish the posterior probability distribution over 7 points.
- ```eigfs_n1200.npy``` contains a numpy array of shape (1200, 10)
- ```Entropy_seeds_delays_clusters.h5``` contains:
  - 'K_range' (9,): values of K that were investigated (1-9)
  - 'entropies' (100, 9, 22): for each (s, K, n) tuple the entropy of the corresponding transition matrix
  - 'n_clusters' (22,): values of n that were investigated (50,  150,  250, ... 2050, 2150)
  - 'seeds' (100,): values of 100 random seeds used
- ```filtered_jmpool_kin.h5```
  - 'MetaData',
  - 'bout_types' (463, 11651): for each bout classifies into one of 13 motor strategies (J_turn, Approach Swim etc.) (1-13 and 15 used for padding).
  - 'converge_bouts' (463, 11651, 175): 
  - 'eye_convergence' (463, 11651): 
  - 'eye_convergence_state' (463, 11651):
  - 'head_pos' (463, 11651, 175, 2): for each of 463 fish, for each of 11651 bouts, for each of 175 frames contains 2-dimensional position of the head
  - 'orientation_smooth' (463, 11651, 175)
  - 'speed_head' (463, 11651, 175): 
  - 'stims' (463, 11651)
  - 'times_bouts' (463, 11651, 2)
- ```kmeans_labels_K5_N1200_s8684.h5```
- ```P_ensemble_ex8_N1200_s8684.npy``` contains the ensemble matrix of shape (1200, 1200)
- ```simlabels_fish_K5_N1200_tau3_cg2.h5```
- ```simlabels_fish_K5_N1200_tau3_cg4.h5```
- ```simlabels_fish_K5_N1200_tau3_cg7.h5```
- ```zebrafish_ms_sims```
