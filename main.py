import yaml 
import wandb
import pprint
import numpy as np
import random
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from train import train_model
from process_data import process_data_and_split
from models.encoder import TransformerEncoder


# Config
with open("config.yaml", "r") as f:
    base_cfg = yaml.safe_load(f)

# wandb init
wandb.init(config=base_cfg)

# Merge sweep dot-keys into nested dict
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
        config[k] = v

pprint.pprint(config)

# Set random seeds for reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data
if config["data"]["process_data_first"]:
    # Process data
    data_splits = process_data_and_split(config, one_hot_encode=config["data"]["one_hot"])
    
    train_data = data_splits['train']
    val_data   = data_splits['val']
    test_data  = data_splits['test']
else:
    # Load the previously processed data
    processed_dir = config["data"]["processed_dir"]

    train_data = torch.load(processed_dir + "/train_.pt")
    val_data   = torch.load(processed_dir + "/val_data.pt")
    test_data  = torch.load(processed_dir + "/test_data.pt")

if config["masking"]["weighted"]:
    # Compute class frequencies for weighted random masking
    print("Using weighted random masking based on class frequencies.")

    flat_labels = train_data["y"].view(-1) # fl
    unique_labels = torch.unique(flat_labels)
    print("Unique labels:", unique_labels.tolist())

    class_frequencies = torch.bincount(flat_labels, minlength=unique_labels.max().item()+1)
    
    for i, f in enumerate(class_frequencies):
        if i in unique_labels:
            print(f"Label {i}: {f} ({f/flat_labels.numel():.4%})")

    # weights should be inversely proportional to the frequencies
    eps = 1e-12
    class_weights = (class_frequencies + eps)**(-1)
    class_weights = class_weights.to(device)
else:
    # Just use uniform random masking
    print("Using uniform random masking.")
    class_weights = None


# extract a single instance for overfitting test
if config["training"]["single_instance_overfit"]:
    print("Using only a single train instance for overfitting test.")
    single_train_instance = train_data[:1]
    train_loader = DataLoader(TensorDataset(single_train_instance), batch_size=1, shuffle=True)
else:
    train_loader = DataLoader(TensorDataset(train_data["x"], train_data["y"]), batch_size=config["training"]["batch_size"], shuffle=True)

val_loader   = DataLoader(TensorDataset(val_data["x"], val_data["y"]) ,   batch_size=config["training"]["batch_size"], shuffle=False)
test_loader  = DataLoader(TensorDataset(test_data["x"], test_data["y"]),  batch_size=config["training"]["batch_size"], shuffle=False)


lr = float(config["training"]["lr"])
print(f"Using learning rate: {lr}")

model = TransformerEncoder(
    input_dim=train_data["x"].shape[-1],
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
reconstruction_loss_name = config["training"]["reconstruction_loss"].lower()
if reconstruction_loss_name == "mse":
    criterion = nn.MSELoss()
elif reconstruction_loss_name == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError(f"Unknown loss function: {config['training']['loss']}")


# Training
print("Starting training...")
history = train_model(
    config,
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    class_weights,
    optimizer, 
    device,
    one_hot_encoded=config["data"]["one_hot"],
)
print("Training complete.")


# Save model
if config["wandb"]["log"]:
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    print(f"Model saved to {model_path}")

wandb.finish()
