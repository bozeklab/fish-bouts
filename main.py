import yaml 
import wandb
import pprint
import numpy as np
import random
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from train import train_model
from process_data import process_data_and_split, one_hot_process_data_and_split
from models.encoder import TransformerEncoder


# --- Load config ---
with open("config.yaml", "r") as f:
    base_cfg = yaml.safe_load(f)

# --- wandb init --- (let the agent set project/entity)
wandb.init(config=base_cfg)

# --- Merge sweep dot-keys into nested dict ---
def set_by_dots(d, dotted_key, value):
    ks = dotted_key.split(".")
    cur = d
    for k in ks[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[ks[-1]] = value

config = base_cfg.copy()
for k, v in dict(wandb.config).items():
    if "." in k:
        set_by_dots(config, k, v)
    else:
        # top-level overrides (if any)
        config[k] = v


# --- Setting random seeds for reproducibility ---
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Prepare data ---
if config["data"]["process_data_first"]:
    if config["data"]["one_hot"]:
        data_splits = one_hot_process_data_and_split(config)
    else:
        data_splits = process_data_and_split(config)
    
    train_data = data_splits['train']
    val_data   = data_splits['val']
    test_data  = data_splits['test']
else:
    processed_dir = config["data"]["processed_dir"]
    train_data = torch.load(processed_dir + "/train_.pt")
    val_data   = torch.load(processed_dir + "/val_data.pt")
    test_data  = torch.load(processed_dir + "/test_data.pt")

if config["masking"]["weighted"]:
    print("Using weighted random masking based on class frequencies.")
    # Compute class weights for weighted random masking
    # assuming train_data is of shape (B, K, C) where C is one-hot
    class_frequencies = train_data.mean(dim=(0, 1))
    eps = 1e-9
    class_weights = (class_frequencies + eps)**(-1)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
else:
    print("Using uniform random masking.")
    class_weights = None

    
print("Train data shape:", train_data.shape)
print("Val data shape:", val_data.shape)
print("Test data shape:", test_data.shape)

# small dataset for overfitting test
if config["training"]["single_instance_overfit"]:
    print("Using only a single train instance for overfitting test.")
    small_train_data = train_data[:1]
    print("Train data shape:", small_train_data.shape)
    train_loader = DataLoader(TensorDataset(small_train_data), batch_size=1, shuffle=True)
else:
    train_loader = DataLoader(TensorDataset(train_data), batch_size=config["training"]["batch_size"], shuffle=True)

val_loader   = DataLoader(TensorDataset(val_data),   batch_size=config["training"]["batch_size"], shuffle=False)
test_loader  = DataLoader(TensorDataset(test_data),  batch_size=config["training"]["batch_size"], shuffle=False)


lr = float(wandb.config["training"]["lr"])
print(f"Using learning rate: {lr}")

model = TransformerEncoder(
    input_dim=train_data.shape[-1],
    seq_len=config["data"]["sequence_length"],
    d_model=config["model"]["d_model"],
    nhead=config["model"]["nhead"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"],
    learnable_mask_embedding=config["masking"]["learnable_mask_embedding"]
).to(device)

print(model)

# Optimizer
opt_name = config["training"]["optimizer"].lower()
if opt_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif opt_name == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
elif opt_name == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:
    raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")

# Loss
loss_name = config["training"]["loss"].lower()
if loss_name == "mse":
    criterion = nn.MSELoss()
elif loss_name == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
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
    class_weights,
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
