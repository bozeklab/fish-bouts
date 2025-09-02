import numpy as np

# data source: https://github.com/GautamSridhar/Markov_Fish/blob/master/FigS10.ipynb

condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','Dark_Transitions(5x5cm)',
                    'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                    'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                    'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']

condition_recs = np.array([[453,463],[121,133],[49,109],[22,49],[163,193],[109,121],
                           [133,164],[443,453],[0,22],
                           [193,258],[304,387],[258,273],[273,304],
                           [387,443]])

conditions = np.zeros((np.max(condition_recs),2),dtype='object')

for k in range(len(condition_recs)):
    t0, tf = condition_recs[k]
    conditions[t0:tf, 0] = np.arange(t0, tf)
    conditions[t0:tf, 1] = [condition_labels[k] for t in range(t0,tf)]


conditions_idx = np.full(np.max(condition_recs), -1, dtype=int)

for idx, (t0, tf) in enumerate(condition_recs):
    conditions_idx[t0:tf] = idx

print("Conditions array shape:", conditions.shape)
print("Conditions array:", conditions)

print("Conditions index shape:", conditions_idx.shape)
print("Conditions index:", conditions_idx)

np.save("sensory_contexts_data.npy", conditions_idx)
