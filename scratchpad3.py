import json
from utils import plot_loss_curve
import matplotlib.pyplot as plt
import numpy as np

with open("training_log.json", "r") as f:
    history = json.load(f)

print(f"Training history loaded with {len(history)} epochs.")
epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")


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
    t0,tf = condition_recs[k]
    conditions[t0:tf,0] = np.arange(t0,tf)
    conditions[t0:tf,1] = [condition_labels[k] for t in range(t0,tf)]


print("Conditions array created with shape:", conditions.shape)
print("Conditions array created with shape:", conditions)


# Create array of -1 initially (meaning "no condition")
conditions_idx = np.full(np.max(condition_recs), -1, dtype=int)

# Fill with condition indices
for idx, (t0, tf) in enumerate(condition_recs):
    conditions_idx[t0:tf] = idx

print("Condition index array:", conditions_idx)
print("Condition index array length:", len(conditions_idx))

counts = np.bincount(conditions_idx, minlength=len(condition_labels))

# Plot histogram
plt.figure(figsize=(10, 5))
plt.bar(range(len(condition_labels)), counts, tick_label=range(len(condition_labels)))
plt.xticks(range(len(condition_labels)), condition_labels, rotation=90)
plt.xlabel('Condition')
plt.ylabel('Count')
plt.title('Histogram of Conditions')
plt.tight_layout()
plt.savefig("conditions_histogram.png")


print("Condition labels:", np.sort(condition_recs, axis=0))
print(condition_recs)
print("Condition counts:", counts)
print("Condition counts sum:", np.sum(counts))

diffs = condition_recs[:, 1] - condition_recs[:, 0]
print("Differences in condition lengths:", diffs)
print("Sum diffs:", np.sum(diffs))