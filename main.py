import torch
import yaml 
import wandb
import pprint
import numpy as np
import random
from torchsummary import summary
from torchviz import make_dot

from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from train import train_model
from process_data import process_data_and_split
from models.encoder import FishBoutEncoder
from models.decoder import FishBoutDecoder


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])


# --- wandb init ---
wandb.init(project="zebrafish", config=config)
config = wandb.config
pprint.pprint(config)


# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Prepare data ---
data_splits = process_data_and_split(config)

train_data = data_splits['train']
val_data   = data_splits['val']
test_data  = data_splits['test']


# small dataset for overfitting test
# small_train_data = train_data[:1]
# print("Train data shape:", small_train_data.shape)
# train_loader = DataLoader(TensorDataset(small_train_data), batch_size=1, shuffle=True)
train_loader = DataLoader(TensorDataset(train_data), batch_size=config["training"]["batch_size"], shuffle=True)
val_loader   = DataLoader(TensorDataset(val_data),   batch_size=config["training"]["batch_size"], shuffle=False)
test_loader  = DataLoader(TensorDataset(test_data),  batch_size=config["training"]["batch_size"], shuffle=False)


lr = float(config["training"]["lr"])  # ensure float
print(f"Using learning rate: {lr}")

model = FishBoutEncoder(
    input_dim=train_data.shape[-1],
    seq_len=config["data"]["sequence_length"],
    d_model=config["model"]["d_model"],
    nhead=config["model"]["nhead"],
    num_layers=config["model"]["num_layers"],
    dim_feedforward=config["model"]["dim_feedforward"],
    dropout=config["model"]["dropout"]
).to(device)

#summary(model, (5,20))
print(model)


if config["training"]["optimizer"].lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif config["training"]["optimizer"].lower() == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
elif config["training"]["optimizer"].lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:
    raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")


if config["training"]["loss"].lower() == "mse":
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Unknown loss function: {config['training']['loss']}")


# --- Training ---
print("Starting training...")
history = train_model(
    config,
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device
)
print("Training complete.")


# --- Save model ---
model_path = "fish_bout_transformer.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
print(f"Model saved to {model_path}")

wandb.finish()

