import torch
import numpy as np
import h5py
import os  
import yaml 
import wandb

from process_data import process_data, load_preprocessed_data, process_data_and_split
from models.encoder import FishBoutEncoder
from models.decoder import FishBoutDecoder

from train import train_model
from evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn
from utils import plot_loss_curve

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

d_model        = config["model"]["d_model"]
nhead          = config["model"]["nhead"]
num_layers     = config["model"]["num_layers"]

batch_size     = config["training"]["batch_size"]
lr             = config["training"]["lr"]
epochs         = config["training"]["epochs"]
optimizer_name = config["training"]["optimizer"]
loss_name      = config["training"]["loss"]

mask_type      = config["masking"]["type"]
mask_percentage= config["masking"]["percentage"]

raw_data_path  = config["data"]["raw_path"]
input_dim      = config["data"]["input_dim"]
seq_len        = config["data"]["sequence_length"]

seed           = config["seed"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Init W&B
wandb.init(
    project="fish-bout-transformer",
    config={
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "optimizer": optimizer_name,
        "loss": loss_name,
        "mask_type": mask_type,
        "mask_percentage": mask_percentage,
        "input_dim": input_dim,
        "seq_len": seq_len
    }
)
config = wandb.config  # Use wandb's tracked config

# -------------------------
# Load or preprocess data
# -------------------------
# processed_path = f'processed/bouts_{input_dim}_{seq_len}.pt'
# if os.path.exists(processed_path):
#     data = load_preprocessed_data(processed_path)
# else:
#     data = process_data(
#         file_path=data_path,
#         n_components=input_dim,
#         window_size=seq_len
#     )

# print(f"Data shape: {data.shape}")
# print(f"Data type: {type(data)}")
# print(f"First 5 samples:\n{data[:5]}")
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


# print("Conditions array created with shape:", conditions.shape)
# print("Conditions array created with shape:", conditions)


# Create array of -1 initially (meaning "no condition")
conditions_idx = np.full(np.max(condition_recs), -1, dtype=int)

# Fill with condition indices
for idx, (t0, tf) in enumerate(condition_recs):
    conditions_idx[t0:tf] = idx

train_fraction = 0.7
val_fraction = 0.15

# num_total = len(data)
# num_train = int(num_total * train_fraction)
# num_val = int(num_total * val_fraction)
# num_test = num_total - num_train - num_val

data_splits = process_data_and_split(
    file_path=raw_data_path,
    conditions_idx=conditions_idx,
    n_components=input_dim,
    window_size=seq_len,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    save_dir="processed",
    seed=seed
)

print("Data splits created with shapes:")
for split_name, tensor in data_splits.items():
    print(f"{split_name}: {tensor.shape}")


# -------------------------
# Split data
# -------------------------


# train_data, val_data, test_data = random_split(data, [num_train, num_val, num_test])
train_data = data_splits['train']
val_data   = data_splits['val']
test_data  = data_splits['test']

train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)


# -------------------------
# Model, optimizer, loss
# -------------------------
print("lr", lr)
lr=float(lr)  # Ensure lr is a float for W&B logging
print(f"Using learning rate: {lr}")
model = FishBoutEncoder(input_dim=train_data.shape[-1], seq_len=seq_len, target_dim=train_data.shape[-1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# -------------------------
# Training
# -------------------------
print("Starting training...")
history = train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    config.epochs, 
    config.mask_type, 
    config.mask_percentage, 
    device,
    wandb_log=True  # <-- Make sure train_model supports W&B logging
)
print("Training complete.")

# Log loss curve
plot_loss_curve(history)
wandb.log({"loss_curve": wandb.Image("loss_curve.png")})  # If plot_loss_curve saves to file

# -------------------------
# Save model
# -------------------------
model_path = "fish_bout_transformer.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
print(f"Model saved to {model_path}")

# -------------------------
# Evaluation
# -------------------------
print("Starting evaluation...")
evaluate_model(model, test_loader, criterion, mask_type, device, wandb_log=True)  # <-- If eval supports logging
print("Evaluation complete.")

wandb.finish()
