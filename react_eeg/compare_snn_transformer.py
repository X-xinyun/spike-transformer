import torch

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
# Seed
# ============================================================

def set_seed(seed=42):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(predictions, labels):

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

    print()
    print("=" * 70)
    print("Building REACT-EEG")
    print("=" * 70)

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
    print("Loading checkpoint:")

    print(
        CHECKPOINT
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown")
    )

    print(
        "Checkpoint F1:",
        checkpoint.get("val_f1", "unknown")
    )

    print("=" * 70)

    return model


# ============================================================
# Compare
# ============================================================

@torch.no_grad()
def compare(
    model,
    loader,
    device
):

    model.eval()

    all_labels = []

    all_snn_predictions = []

    all_transformer_predictions = []

    all_final_predictions = []

    # ========================================================
    # Important counters
    # ========================================================

    snn_wrong_transformer_correct = 0

    snn_correct_transformer_wrong = 0

    both_wrong = 0

    both_correct = 0

    seizure_snn_wrong_transformer_correct = 0

    seizure_snn_wrong_transformer_wrong = 0

    total_samples = 0

    # ========================================================
    # Loop
    # ========================================================

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        out = model(x)

        snn_logits = out["snn_logits"]

        transformer_logits = out["transformer_logits"]

        final_logits = out["logits"]

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        snn_prediction = torch.argmax(
            snn_logits,
            dim=1
        )

        transformer_prediction = torch.argmax(
            transformer_logits,
            dim=1
        )

        final_prediction = torch.argmax(
            final_logits,
            dim=1
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_labels.append(
            y.cpu()
        )

        all_snn_predictions.append(
            snn_prediction.cpu()
        )

        all_transformer_predictions.append(
            transformer_prediction.cpu()
        )

        all_final_predictions.append(
            final_prediction.cpu()
        )

        # ----------------------------------------------------
        # Correct / wrong
        # ----------------------------------------------------

        snn_correct = (
            snn_prediction == y
        )

        transformer_correct = (
            transformer_prediction == y
        )

        # ----------------------------------------------------
        # SNN wrong
        # Transformer correct
        # ----------------------------------------------------

        condition_1 = (
            (~snn_correct)
            &
            transformer_correct
        )

        snn_wrong_transformer_correct += (
            condition_1.sum().item()
        )

        # ----------------------------------------------------
        # SNN correct
        # Transformer wrong
        # ----------------------------------------------------

        condition_2 = (
            snn_correct
            &
            (~transformer_correct)
        )

        snn_correct_transformer_wrong += (
            condition_2.sum().item()
        )

        # ----------------------------------------------------
        # Both wrong
        # ----------------------------------------------------

        condition_3 = (
            (~snn_correct)
            &
            (~transformer_correct)
        )

        both_wrong += (
            condition_3.sum().item()
        )

        # ----------------------------------------------------
        # Both correct
        # ----------------------------------------------------

        condition_4 = (
            snn_correct
            &
            transformer_correct
        )

        both_correct += (
            condition_4.sum().item()
        )

        # ----------------------------------------------------
        # Important:
        #
        # seizure samples where:
        #
        # SNN = wrong
        # Transformer = correct
        # ----------------------------------------------------

        seizure_condition_1 = (
            (y == 1)
            &
            (~snn_correct)
            &
            transformer_correct
        )

        seizure_snn_wrong_transformer_correct += (
            seizure_condition_1.sum().item()
        )

        # ----------------------------------------------------
        # seizure samples where both are wrong
        # ----------------------------------------------------

        seizure_condition_2 = (
            (y == 1)
            &
            (~snn_correct)
            &
            (~transformer_correct)
        )

        seizure_snn_wrong_transformer_wrong += (
            seizure_condition_2.sum().item()
        )

        total_samples += x.size(0)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx % 100 == 0
            or batch_idx == len(loader) - 1
        ):

            print(
                f"Batch "
                f"{batch_idx + 1:5d}/"
                f"{len(loader):5d}"
            )

    # ========================================================
    # Concatenate
    # ========================================================

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    all_snn_predictions = torch.cat(
        all_snn_predictions,
        dim=0
    )

    all_transformer_predictions = torch.cat(
        all_transformer_predictions,
        dim=0
    )

    all_final_predictions = torch.cat(
        all_final_predictions,
        dim=0
    )

    # ========================================================
    # Metrics
    # ========================================================

    snn_metrics = calculate_metrics(
        all_snn_predictions,
        all_labels
    )

    transformer_metrics = calculate_metrics(
        all_transformer_predictions,
        all_labels
    )

    final_metrics = calculate_metrics(
        all_final_predictions,
        all_labels
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 70)
    print("SNN vs Transformer Comparison")
    print("=" * 70)

    # --------------------------------------------------------
    # SNN
    # --------------------------------------------------------

    print()
    print("SNN ONLY")
    print("-" * 70)

    print(
        f"Accuracy:    "
        f"{snn_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Sensitivity: "
        f"{snn_metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{snn_metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"Precision:   "
        f"{snn_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"F1:          "
        f"{snn_metrics['f1']:.4f}"
    )

    print(
        f"TP: {snn_metrics['tp']} | "
        f"TN: {snn_metrics['tn']} | "
        f"FP: {snn_metrics['fp']} | "
        f"FN: {snn_metrics['fn']}"
    )

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    print()
    print("TRANSFORMER ONLY")
    print("-" * 70)

    print(
        f"Accuracy:    "
        f"{transformer_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Sensitivity: "
        f"{transformer_metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{transformer_metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"Precision:   "
        f"{transformer_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"F1:          "
        f"{transformer_metrics['f1']:.4f}"
    )

    print(
        f"TP: {transformer_metrics['tp']} | "
        f"TN: {transformer_metrics['tn']} | "
        f"FP: {transformer_metrics['fp']} | "
        f"FN: {transformer_metrics['fn']}"
    )

    # --------------------------------------------------------
    # Final REACT-EEG
    # --------------------------------------------------------

    print()
    print("FINAL REACT-EEG")
    print("-" * 70)

    print(
        f"Accuracy:    "
        f"{final_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Sensitivity: "
        f"{final_metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{final_metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"Precision:   "
        f"{final_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"F1:          "
        f"{final_metrics['f1']:.4f}"
    )

    print(
        f"TP: {final_metrics['tp']} | "
        f"TN: {final_metrics['tn']} | "
        f"FP: {final_metrics['fp']} | "
        f"FN: {final_metrics['fn']}"
    )

    # ========================================================
    # Error relationship
    # ========================================================

    print()
    print("=" * 70)
    print("Error Relationship")
    print("=" * 70)

    print()
    print(
        "SNN wrong -> Transformer correct:"
    )

    print(
        f"    {snn_wrong_transformer_correct}"
    )

    print()
    print(
        "SNN correct -> Transformer wrong:"
    )

    print(
        f"    {snn_correct_transformer_wrong}"
    )

    print()
    print(
        "Both wrong:"
    )

    print(
        f"    {both_wrong}"
    )

    print()
    print(
        "Both correct:"
    )

    print(
        f"    {both_correct}"
    )

    # ========================================================
    # Seizure-specific analysis
    # ========================================================

    print()
    print("=" * 70)
    print("Seizure-specific Error Analysis")
    print("=" * 70)

    total_seizures = (
        (all_labels == 1)
        .sum()
        .item()
    )

    print()
    print(
        "Total seizure samples:"
    )

    print(
        f"    {total_seizures}"
    )

    print()
    print(
        "Seizure: SNN wrong -> Transformer correct:"
    )

    print(
        f"    {seizure_snn_wrong_transformer_correct}"
    )

    print()
    print(
        "Seizure: both SNN and Transformer wrong:"
    )

    print(
        f"    {seizure_snn_wrong_transformer_wrong}"
    )

    # ========================================================
    # Interpretation
    # ========================================================

    print()
    print("=" * 70)
    print("Interpretation")
    print("=" * 70)

    if (
        seizure_snn_wrong_transformer_correct > 0
    ):

        print()
        print(
            "GOOD:"
        )

        print(
            "Transformer can correct "
            "some SNN seizure errors."
        )

        print()
        print(
            "This means the Transformer "
            "has useful complementary information."
        )

    else:

        print()
        print(
            "WARNING:"
        )

        print(
            "Transformer did not correct "
            "any SNN seizure errors."
        )

        print()
        print(
            "We should NOT tune the Router yet."
        )

        print(
            "We should first investigate "
            "Transformer training."
        )

    print()
    print("=" * 70)
    print("Comparison finished")
    print("=" * 70)


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
    print("REACT-EEG SNN / Transformer Analysis")
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
    # Compare
    # ========================================================

    compare(
        model=model,
        loader=val_loader,
        device=device
    )


if __name__ == "__main__":

    main()

