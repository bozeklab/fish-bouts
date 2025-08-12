import torch
from utils import apply_mask
from torch.utils.data import DataLoader, TensorDataset
import json

def evaluate_model(model, dataloader, criterion, mask_type, device, log_path="evaluation_results.json"):
    model.eval()
    total_loss = 0.0
    batch_losses = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, batch in enumerate(dataloader, start=1):
            x = batch[0].to(device)

            x_masked, mask = apply_mask(x, mask_type=mask_type)

            outputs = model(src=x_masked, tgt=x)

            # Loss only on masked positions
            loss = criterion(outputs[mask], x[mask])
            batch_losses.append(loss.item())
            total_loss += loss.item()

            print(f"Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    average_loss = total_loss / len(dataloader)

    # Prepare results
    results = {
        "average_loss": average_loss,
        "num_batches": len(dataloader),
        "batch_losses": batch_losses
    }

    # Save to file
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print summary
    print(f"\nEvaluation complete.")
    print(f"Average loss over {len(dataloader)} batches: {average_loss:.4f}")
    print(f"Results saved to {log_path}")

    return average_loss
