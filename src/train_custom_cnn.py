import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    use_amp=True,
    amp_dtype=torch.bfloat16
):
    model.train()

    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(train_loader)
    elapsed = time.time() - start_time

    return epoch_loss, elapsed
def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    labels,
    use_amp=True,
    amp_dtype=torch.bfloat16
):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs)

            all_probs.append(probabilities.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    # Combine predictions from all validation batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate AUROC separately for each disease
    per_label_auroc = {}

    for i, label in enumerate(labels):

        # AUROC requires both positive and negative examples
        if len(np.unique(all_labels[:, i])) == 2:
            auc = roc_auc_score(
                all_labels[:, i],
                all_probs[:, i]
            )
            per_label_auroc[label] = auc
        else:
            per_label_auroc[label] = float("nan")

    macro_auroc = np.nanmean(list(per_label_auroc.values()))

    val_loss = running_loss / len(val_loader)

    return val_loss, macro_auroc, per_label_auroc