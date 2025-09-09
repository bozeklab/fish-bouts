import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from utils import apply_mask

def train_model(config, 
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                device
    ):
    model.to(device)

    num_epochs=config["training"]["epochs"]
    mask_type=config["masking"]["type"]
    mask_ratio=config["masking"]["percentage"]

    wandb_log=True

    history = []  # store {"epoch": X, "train_loss": Y, "val_loss": Z, "train_acc": A, "val_acc": B}

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0
        train_preds_all = []
        train_targets_all = []


        for batch in train_loader:
            # print(f"{batch=}")
            x = batch[0].to(device)
            # print(f"{x.shape=}")
            # print(f"{x=}")
            x_masked, mask = apply_mask(x, mask_type=mask_type, mask_ratio=mask_ratio, if_print=True)
            x_masked = x_masked.to(device)
            mask = mask.to(device)
            # print(f"{x_masked.shape=}")
            # print(f"{x_masked=}")
            # print(f"{mask.shape=}")
            output = model(src=x_masked, mask_positions=mask)
            # print(f"{output.shape=}")
            # print(f"{output=}")
            # Loss only at masked positions
            loss = criterion(output[mask], x[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # ---- Accuracy ----
            # print(f"{output.shape=}")
            # print(f"{output[mask].shape=}")
            # print(f"{x.shape=}")
            # print(f"{x[mask].shape=}")
            preds = torch.argmax(output, dim=-1)
            targets = torch.argmax(x, dim=-1)
            correct = (preds[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
            train_correct += correct
            train_total += total

            if total > 0:
                train_preds_all.append(preds[mask].detach().cpu())
                train_targets_all.append(targets[mask].detach().cpu())


        avg_train_loss = train_loss / train_batches
        train_acc = train_correct / train_total if train_total > 0 else 0.0


        num_classes = output.size(-1)

        # print(f"{train_preds_all=}")
        # print(f"{train_targets_all=}")
        if train_preds_all:
            train_preds_all = torch.cat(train_preds_all).numpy()
            train_targets_all = torch.cat(train_targets_all).numpy()
            train_f1 = f1_score(
                train_targets_all, train_preds_all,
                average="macro",
                labels=list(range(num_classes)),
                zero_division=0
            )

        else:
            train_f1 = 0.0

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)

                x_masked, mask = apply_mask(x, mask_type=mask_type, mask_ratio=mask_ratio, if_print=False)
                x_masked = x_masked.to(device)
                mask = mask.to(device)
                output = model(src=x_masked, mask_positions=mask)
                
                loss = criterion(output[mask], x[mask])
                val_loss += loss.item()
                val_batches += 1

                # ---- Accuracy ----
                # print(f"{output.shape=}")
                # print(f"{output[mask].shape=}")
                # print(f"{x.shape=}")
                # print(f"{x[mask].shape=}")
                preds = torch.argmax(output, dim=-1)
                targets = torch.argmax(x, dim=-1)
                correct = (preds[mask] == targets[mask]).sum().item()
                total = mask.sum().item()
                val_correct += correct
                val_total += total

                # ---- Collect for F1 (masked positions only) ----
                if total > 0:
                    val_preds_all.append(preds[mask].detach().cpu())
                    val_targets_all.append(targets[mask].detach().cpu())

        avg_val_loss = val_loss / val_batches
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if val_preds_all:
            val_preds_all = torch.cat(val_preds_all).numpy()
            val_targets_all = torch.cat(val_targets_all).numpy()
            val_f1 = f1_score(
                val_targets_all, val_preds_all,
                average="macro",
                labels=list(range(num_classes)),
                zero_division=0
            )
        else:
            val_f1 = 0.0


        # ---- Logging ----
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Train F1 (macro): {train_f1:.4f} | Val F1 (macro): {val_f1:.4f}")

        if wandb_log:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_f1": train_f1,
                "val_f1": val_f1,
            })

        #---- Early Stopping Check ----
        if config["training"]["early_stopping"]["enabled"]:
            patience = config["training"]["early_stopping"]["patience"]
            delta = float(config["training"]["early_stopping"]["delta"])
            if avg_val_loss < best_val_loss - delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_best_path=f"best_model.pth"
                torch.save(model.state_dict(), save_best_path)
                print(f"Validation loss improved. Model saved to {save_best_path}")

                # --- Confusion matrix ---
                if len(val_preds_all) > 0:
                    cm = confusion_matrix(
                        val_targets_all, val_preds_all,
                        labels=list(range(num_classes))
                    )
                    plt.figure(figsize=(8, 6))
                    
                    plt.imshow(cm, aspect="auto")
                    plt.colorbar()
                    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.xticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))
                    plt.yticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))

                    for i in range(num_classes):
                        for j in range(num_classes):
                            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
                    if wandb_log:
                        wandb.log({"confusion_matrix_best": wandb.Image(plt), "best_epoch": epoch + 1})
                    plt.savefig("confusion_matrix_best.png")
                    print("Confusion matrix saved as confusion_matrix_best.png")
                    plt.close()
            else:
                patience_counter += 1
                print(f"No improvement. Early stopping patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

    return history
