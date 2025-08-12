import torch
import math
import json

from torch import nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset

from process_data import process_data, load_preprocessed_data
from models.transformer import FishBoutBERT
from utils import apply_mask


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, mask_type, mask_ratio, device, log_path="training_log.json"):
    model.to(device)
    history = []  # store {"epoch": X, "train_loss": Y, "val_loss": Z}

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)

            x_masked, mask = apply_mask(x, mask_type=mask_type, mask_ratio=mask_ratio)
            x_masked = x_masked.to(device)
            mask = mask.to(device)
            output = model(src=x_masked, mask_positions=mask)

            # Loss only at masked positions
            loss = criterion(output[mask], x[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)

                x_masked, mask = apply_mask(x, mask_type=mask_type, mask_ratio=mask_ratio)
                x_masked = x_masked.to(device)
                mask = mask.to(device)
                output = model(src=x_masked, mask_positions=mask)

                loss = criterion(output[mask], x[mask])
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # ---- Logging ----
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save to JSON file
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Training log saved to {log_path}")
    return history