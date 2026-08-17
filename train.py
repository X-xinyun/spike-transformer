import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from react_eeg.data.chbmit_dataset import CHBMITDataset
from react_eeg.models.react_eeg import REACTEEG
from spikingjelly.clock_driven import functional

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = r"D:\Qiang\数据集\chb-mit-scalp-eeg-database-1.0.0"

PATIENT = "chb01"

BATCH_SIZE = 4

EPOCHS = 1

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

TRAIN_RATIO = 0.8

VAL_RATIO = 0.2

SEED = 42


# ============================================================
# Random seed
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Calculate classification metrics
# ============================================================

def calculate_metrics(logits, labels):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    predictions = predictions.detach().cpu()

    labels = labels.detach().cpu()

    tp = ((predictions == 1) & (labels == 1)).sum().item()

    tn = ((predictions == 0) & (labels == 0)).sum().item()

    fp = ((predictions == 1) & (labels == 0)).sum().item()

    fn = ((predictions == 0) & (labels == 1)).sum().item()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * sensitivity
        / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(labels)
        if len(labels) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# Build dataloader
# ============================================================

def build_dataset():

    print()
    print("=" * 70)
    print("Loading CHB-MIT dataset")
    print("=" * 70)

    dataset = CHBMITDataset(
        root_dir=DATA_ROOT,
        patient=PATIENT,
        window_seconds=5.0,
        sampling_rate=256,
        stride_seconds=5.0,
        normalize=True
    )

    return dataset


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch
):

    model.train()

    total_loss = 0.0

    total_samples = 0

    all_logits = []

    all_labels = []

    total_trigger = 0

    print()
    print("=" * 70)
    print(f"Training Epoch {epoch}")
    print("=" * 70)

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        out = model(x)

        final_logits = out["logits"]

        snn_logits = out["snn_logits"]

        transformer_logits = out["transformer_logits"]

        trigger = out["trigger"]

        # --------------------------------------------------------
        # Loss
        # --------------------------------------------------------

        final_loss = criterion(
            final_logits,
            y
        )

        snn_loss = criterion(
            snn_logits,
            y
        )

        transformer_loss = criterion(
            transformer_logits,
            y
        )
                # --------------------------------------------------------
        # Risk supervision
        #
        # risk:
        #     [B]
        #
        # y:
        #     [B]
        #
        # We directly teach the SNN risk head:
        #
        #     non-seizure -> risk = 0
        #     seizure     -> risk = 1
        #
        # Because CHB-MIT is highly imbalanced, use a
        # positive-class weight.
        # --------------------------------------------------------

        risk = out["risk"]

        positive_weight = 50.0

        risk_loss_fn = nn.BCELoss(
            weight=torch.where(
                y == 1,
                torch.tensor(
                    positive_weight,
                    device=device
                ),
                torch.tensor(
                    1.0,
                    device=device
                )
            )
        )

        risk_loss = risk_loss_fn(
            risk,
            y.float()
        )

        # --------------------------------------------------------
        # First-version total loss
        # --------------------------------------------------------

        loss = (
            0.40 * final_loss
            +
            0.20 * snn_loss
            +
            0.20 * transformer_loss
            +
            0.20 * risk_loss
        )

        # --------------------------------------------------------
        # Backward
        # --------------------------------------------------------

        loss.backward()

        optimizer.step()

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        batch_size = x.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_logits.append(
            final_logits.detach().cpu()
        )

        all_labels.append(
            y.detach().cpu()
        )

        total_trigger += (
            trigger.float().sum().item()
        )

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if (
            batch_idx % 50 == 0
            or batch_idx == len(loader) - 1
        ):

            current_loss = (
                total_loss / total_samples
            )

            trigger_rate = (
                total_trigger / total_samples
            )

            print(
                f"Batch "
                f"{batch_idx + 1:5d}/"
                f"{len(loader):5d} | "
                f"Loss: "
                f"{current_loss:.6f} | "
                f"Trigger: "
                f"{trigger_rate * 100:.2f}%"
            )

    # ------------------------------------------------------------
    # Epoch metrics
    # ------------------------------------------------------------

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    metrics = calculate_metrics(
        all_logits,
        all_labels
    )

    average_loss = (
        total_loss / total_samples
    )

    trigger_rate = (
        total_trigger / total_samples
    )

    print()
    print("-" * 70)
    print("Training Result")
    print("-" * 70)

    print(
        f"Loss:        {average_loss:.6f}"
    )

    print(
        f"Accuracy:    {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Sensitivity: {metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity: {metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"Precision:   {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"F1:          {metrics['f1']:.4f}"
    )

    print(
        f"Trigger:     {trigger_rate * 100:.2f}%"
    )

    print("-" * 70)

    return average_loss, metrics


# ============================================================
# Validation
# ============================================================

# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_labels = []

    all_risk = []
    all_uncertainty = []
    all_evidence = []
    all_reliability = []

    total_trigger = 0

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        out = model(x)

        logits = out["logits"]

        trigger = out["trigger"]

        risk = out["risk"]

        uncertainty = out["uncertainty"]

        evidence = out["evidence"]

        mean_reliability = out["mean_reliability"]

        # --------------------------------------------------------
        # Loss
        # --------------------------------------------------------

        loss = criterion(
            logits,
            y
        )

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        batch_size = x.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            y.cpu()
        )

        total_trigger += (
            trigger.float().sum().item()
        )

        all_risk.append(
            risk.detach().cpu()
        )

        all_uncertainty.append(
            uncertainty.detach().cpu()
        )

        all_evidence.append(
            evidence.detach().cpu()
        )

        all_reliability.append(
            mean_reliability.detach().cpu()
        )

    # ------------------------------------------------------------
    # Concatenate all batches
    # ------------------------------------------------------------

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    all_risk = torch.cat(
        all_risk,
        dim=0
    )

    all_uncertainty = torch.cat(
        all_uncertainty,
        dim=0
    )

    all_evidence = torch.cat(
        all_evidence,
        dim=0
    )

    all_reliability = torch.cat(
        all_reliability,
        dim=0
    )

    # ------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        all_logits,
        all_labels
    )

    average_loss = (
        total_loss / total_samples
    )

    trigger_rate = (
        total_trigger / total_samples
    )

    # ------------------------------------------------------------
    # Evidence statistics
    # ------------------------------------------------------------

    mean_risk = all_risk.mean().item()

    mean_uncertainty = (
        all_uncertainty.mean().item()
    )

    mean_evidence = (
        all_evidence.mean().item()
    )

    mean_reliability = (
        all_reliability.mean().item()
    )

    # ------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("Validation Result")
    print("=" * 70)

    print(
        f"Loss:             {average_loss:.6f}"
    )

    print(
        f"Accuracy:         {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Sensitivity:      {metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity:      {metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"Precision:        {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"F1:               {metrics['f1']:.4f}"
    )

    print(
        f"Trigger rate:     {trigger_rate * 100:.2f}%"
    )

    print(
        f"Mean risk:        {mean_risk:.4f}"
    )

    print(
        f"Mean uncertainty: {mean_uncertainty:.4f}"
    )

    print(
        f"Mean evidence:    {mean_evidence:.4f}"
    )

    print(
        f"Mean reliability: {mean_reliability:.4f}"
    )

    print(
        f"TP: {metrics['tp']} | "
        f"TN: {metrics['tn']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']}"
    )

    print("=" * 70)

    return average_loss, metrics


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("REACT-EEG Training")
    print("=" * 70)

    print(
        "Device:",
        device
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Patient:",
        PATIENT
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = build_dataset()

    # --------------------------------------------------------
    # Train / validation split
    #
    # IMPORTANT:
    # This is only the first runnable version.
    # Later we will replace this with patient-independent
    # and file-aware splitting.
    # --------------------------------------------------------

    train_size = int(
        len(dataset) * TRAIN_RATIO
    )

    val_size = (
        len(dataset) - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print()
    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = REACTEEG(
        in_channels=22,
        time_steps=4,
        snn_hidden_channels=32,
        transformer_channels=8,
        num_classes=2,
        embed_dim=128,
        num_heads=8,
        num_layers=2
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    num_seizure = sum(
      dataset.samples[i]["label"] == 1
      for i in range(len(dataset))
    )

    num_non_seizure = (
      len(dataset) - num_seizure
    )

    seizure_weight = min(
     num_non_seizure / max(num_seizure, 1),
     20.0
     )

    class_weights = torch.tensor(
      [1.0, seizure_weight],
      dtype=torch.float32,
      device=device
    )

    print()
    print("=" * 70)
    print("Class imbalance")
    print("=" * 70)

    print(f"Non-seizure samples: {num_non_seizure}")
    print(f"Seizure samples:     {num_seizure}")
    print(f"Seizure weight:       {seizure_weight:.4f}")

    print("=" * 70)

    criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        [1.0, 20.0],
        dtype=torch.float32,
        device=device
    )
)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Save directory
    # --------------------------------------------------------

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    best_f1 = -1.0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            checkpoint_path = (
                "checkpoints/"
                "react_eeg_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                    "val_sensitivity": val_metrics["sensitivity"],
                    "val_specificity": val_metrics["specificity"],
                },
                checkpoint_path
            )

            print()
            print(
                "Best model saved:"
            )

            print(
                checkpoint_path
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)

    print(
        "Best validation F1:",
        best_f1
    )


if __name__ == "__main__":

    main()