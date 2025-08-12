import torch
import numpy as np
import h5py
import os  
import yaml 

from process_data import process_data, load_preprocessed_data, compute_statistics
from models.transformer import FishBoutBERT
from train import train_model
from evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn

from utils import plot_loss_curve

# Load hyperparameters from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access config values
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

data_path      = config["data"]["path"]
input_dim      = config["data"]["input_dim"]
seq_len        = config["data"]["sequence_length"]

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

if os.path.exists(f'processed/bouts_{input_dim}_{seq_len}.pt'):
    data = load_preprocessed_data(f'processed/bouts_{input_dim}_{seq_len}.pt')
else:
    data = process_data(
        file_path='Datasets/JM_data/pool_ex8_PCs.h5',
        n_components=input_dim,
        window_size=window_size
    )

print(f"Data shape: {data.shape}")  # Should be (N_windows, 100) where N_windows is variable
print(f"Data type: {type(data)}")  # Should be torch.Tensor
print(f"First 5 samples:\n{data[:5]}")  # Print first 5 samples for verification

#compute_statistics(data)  # Compute and print statistics of the data

train_fraction = 0.7
val_fraction = 0.15

num_total = len(data)
num_train = int(num_total * train_fraction)
num_val = int(num_total * val_fraction)
num_test = num_total - num_train - num_val

train_data, val_data, test_data = random_split(data, [num_train, num_val, num_test])

train_loader = DataLoader(
    TensorDataset(train_data.dataset[train_data.indices]),
    batch_size=64,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(val_data.dataset[val_data.indices]),
    batch_size=64,
    shuffle=False
)
test_loader = DataLoader(
    TensorDataset(test_data.dataset[test_data.indices]),
    batch_size=64,
    shuffle=False
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

print(f"Train first batch: {next(iter(train_loader))[0].shape}")
print(f"Val first batch: {next(iter(val_loader))[0].shape}")
print(f"Test first batch: {next(iter(test_loader))[0].shape}")


model = FishBoutBERT(input_dim=data.shape[-1], seq_len=seq_len, target_dim=data.shape[-1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 5
mask_type = 'random'

print("Starting training...")
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, mask_type, mask_percentage, device)
print("Training complete.")

plot_loss_curve(history)

model_path = "fish_bout_transformer.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


model.load_state_dict(torch.load("fish_bout_transformer.pth", map_location='cpu'))

print("Starting evaluation...")
evaluate_model(model, test_loader, criterion, mask_type, device)
print("Evaluation complete.")