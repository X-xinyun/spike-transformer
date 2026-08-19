import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from react_eeg.data.chbmit_dataset import CHBMITDataset
from react_eeg.models.react_eeg import REACTEEG


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = r"D:\Qiang\数据集\chb-mit-scalp-eeg-database-1.0.0"

PATIENT = "chb01"

CHECKPOINT = r"checkpoints\react_eeg_best.pth"

BATCH_SIZE = 4

NUM_WORKERS = 0

TRAIN_RATIO = 0.8

SEED = 42


# ============================================================
# Router thresholds to test
# ============================================================

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# Random seed
# ============================================================

def set_seed(seed=42):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(logits, labels):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    predictions = predictions.cpu()

    labels = labels.cpu()

    tp = (
        (predictions == 1)
        &
        (labels == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        &
        (labels == 0)
    ).sum().item()

    fp = (
        (predictions == 1)
        &
        (labels == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        &
        (labels == 1)
    ).sum().item()

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
        2.0
        * precision
        * sensitivity
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
# Build dataset
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
# Build model
# ============================================================

def build_model(device):

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

    print()
    print("=" * 70)
    print("Loading checkpoint")
    print("=" * 70)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown")
    )

    print(
        "Checkpoint validation F1:",
        checkpoint.get("val_f1", "unknown")
    )

    print("=" * 70)

    return model


# ============================================================
# Evaluate one threshold
# ============================================================

@torch.no_grad()
def evaluate_threshold(
    model,
    loader,
    threshold,
    device
):

    # --------------------------------------------------------
    # Change only Router threshold
    # --------------------------------------------------------

    model.router.trigger_threshold = threshold

    model.eval()

    all_logits = []

    all_labels = []

    total_trigger = 0

    total_samples = 0

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        out = model(x)

        logits = out["logits"]

        trigger = out["trigger"]

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            y.cpu()
        )

        total_trigger += (
            trigger.float().sum().item()
        )

        total_samples += x.size(0)

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        all_logits,
        all_labels
    )

    trigger_rate = (
        total_trigger / total_samples
    )

    return metrics, trigger_rate


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
    print("REACT-EEG Router Calibration")
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
        "Checkpoint:",
        CHECKPOINT
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = build_dataset()

    train_size = int(
        len(dataset) * TRAIN_RATIO
    )

    val_size = (
        len(dataset) - train_size
    )

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print()
    print(
        "Validation samples:",
        len(val_dataset)
    )

    # ========================================================
    # DataLoader
    # ========================================================

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda")
    )

    # ========================================================
    # Model
    # ========================================================

    model = build_model(device)

    # ========================================================
    # Calibration
    # ========================================================

    print()
    print("=" * 100)
    print("Threshold Calibration")
    print("=" * 100)

    print(
        f"{'Threshold':>10} | "
        f"{'Trigger %':>10} | "
        f"{'Accuracy %':>11} | "
        f"{'Sensitivity %':>14} | "
        f"{'Specificity %':>14} | "
        f"{'Precision %':>12} | "
        f"{'F1':>8} | "
        f"{'TP':>5} | "
        f"{'FP':>5} | "
        f"{'FN':>5}"
    )

    print("-" * 100)

    results = []

    for threshold in THRESHOLDS:

        metrics, trigger_rate = evaluate_threshold(
            model=model,
            loader=val_loader,
            threshold=threshold,
            device=device
        )

        result = {
            "threshold": threshold,
            "trigger_rate": trigger_rate,
            **metrics
        }

        results.append(result)

        print(
            f"{threshold:10.2f} | "
            f"{trigger_rate * 100:10.2f} | "
            f"{metrics['accuracy'] * 100:11.2f} | "
            f"{metrics['sensitivity'] * 100:14.2f} | "
            f"{metrics['specificity'] * 100:14.2f} | "
            f"{metrics['precision'] * 100:12.2f} | "
            f"{metrics['f1']:8.4f} | "
            f"{metrics['tp']:5d} | "
            f"{metrics['fp']:5d} | "
            f"{metrics['fn']:5d}"
        )

    print("=" * 100)

    # ========================================================
    # Find best thresholds
    # ========================================================

    print()
    print("=" * 70)
    print("Calibration Summary")
    print("=" * 70)

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1 = max(
        results,
        key=lambda x: x["f1"]
    )

    print()
    print("Best F1 threshold:")

    print(
        f"Threshold:   {best_f1['threshold']:.2f}"
    )

    print(
        f"Sensitivity: {best_f1['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity: {best_f1['specificity'] * 100:.2f}%"
    )

    print(
        f"F1:          {best_f1['f1']:.4f}"
    )

    print(
        f"Trigger:     {best_f1['trigger_rate'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Best sensitivity with minimum trigger rate
    #
    # Target sensitivity:
    #     90%
    #
    # If no threshold reaches 90%,
    # use the threshold with highest sensitivity.
    # --------------------------------------------------------

    target_sensitivity = 0.90

    feasible = [
        r
        for r in results
        if r["sensitivity"] >= target_sensitivity
    ]

    print()
    print(
        f"Target sensitivity: "
        f"{target_sensitivity * 100:.1f}%"
    )

    if len(feasible) > 0:

        best_cost = min(
            feasible,
            key=lambda x: x["trigger_rate"]
        )

        print()
        print(
            "Best threshold satisfying "
            "target sensitivity:"
        )

        print(
            f"Threshold:   {best_cost['threshold']:.2f}"
        )

        print(
            f"Sensitivity: "
            f"{best_cost['sensitivity'] * 100:.2f}%"
        )

        print(
            f"Specificity: "
            f"{best_cost['specificity'] * 100:.2f}%"
        )

        print(
            f"Precision:   "
            f"{best_cost['precision'] * 100:.2f}%"
        )

        print(
            f"F1:          "
            f"{best_cost['f1']:.4f}"
        )

        print(
            f"Trigger rate:"
            f" {best_cost['trigger_rate'] * 100:.2f}%"
        )

        print()
        print(
            "This threshold is the first "
            "candidate for REACT-EEG deployment."
        )

    else:

        best_sensitivity = max(
            results,
            key=lambda x: x["sensitivity"]
        )

        print()
        print(
            "No threshold reached the "
            "target sensitivity."
        )

        print()
        print(
            "Highest sensitivity threshold:"
        )

        print(
            f"Threshold:   "
            f"{best_sensitivity['threshold']:.2f}"
        )

        print(
            f"Sensitivity: "
            f"{best_sensitivity['sensitivity'] * 100:.2f}%"
        )

        print(
            f"Trigger rate:"
            f" {best_sensitivity['trigger_rate'] * 100:.2f}%"
        )

    print()
    print("=" * 70)
    print("Router calibration finished")
    print("=" * 70)


if __name__ == "__main__":

    main()

