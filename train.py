import math
import json
import wandb
import torch

from torch import nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset

from process_data import process_data, load_preprocessed_data
from models.encoder import FishBoutEncoder
from utils import apply_mask


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    mask_type,
    mask_ratio,
    device,
    log_path="training_log.json",
    wandb_log=False,
    patience=10,
    delta=0.0001,
    save_best_path="best_model.pth"

):
    model.to(device)
    history = []  # store {"epoch": X, "train_loss": Y, "val_loss": Z}

    best_val_loss = float("inf")
    patience_counter = 0

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

        if wandb_log:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

        # ---- Early Stopping Check ----
        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_best_path)
            print(f"Validation loss improved. Model saved to {save_best_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Early stopping patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Save to JSON file
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Training log saved to {log_path}")
    return history
