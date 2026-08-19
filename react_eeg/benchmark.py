import os
import torch

from torch.utils.data import DataLoader

from react_eeg.data.chbmit_dataset import CHBMITDataset
from react_eeg.models.react_eeg import REACTEEG
from spikingjelly.clock_driven import functional


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = r"D:\Qiang\数据集\chb-mit-scalp-eeg-database-1.0.0"

PATIENT = "chb01"

CHECKPOINT_PATH = r"checkpoints\react_eeg_best.pth"

BATCH_SIZE = 4

NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(logits, labels):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    predictions = predictions.detach().cpu()
    labels = labels.detach().cpu()

    tp = (
        (predictions == 1)
        & (labels == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        & (labels == 0)
    ).sum().item()

    fp = (
        (predictions == 1)
        & (labels == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        & (labels == 1)
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
# Print metrics
# ============================================================

def print_metrics(title, metrics):

    print()
    print(title)
    print("-" * 70)

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
        f"TP: {metrics['tp']} | "
        f"TN: {metrics['tn']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']}"
    )


# ============================================================
# Reset SNN states
# ============================================================

def reset_network(model):

    try:
        functional.reset_net(model)
    except Exception:
        pass


# ============================================================
# Load dataset
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

    print(
        f"Total samples: {len(dataset)}"
    )

    return dataset


# ============================================================
# Load model
# ============================================================

def build_model():

    print()
    print("=" * 70)
    print("Loading REACT-EEG model")
    print("=" * 70)

    # IMPORTANT:
    # Current REACT-EEG uses 22 EEG channels.
    #
    # Do NOT change transformer_channels to 8.
    #
    model = REACTEEG(
        in_channels=22,
        time_steps=4,
        snn_hidden_channels=32,
        transformer_channels=22,
        num_classes=2,
        embed_dim=128,
        num_heads=8,
        num_layers=2
    ).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint loaded: {CHECKPOINT_PATH}"
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "val_f1" in checkpoint:

        print(
            f"Checkpoint validation F1: "
            f"{checkpoint['val_f1']:.4f}"
        )

    model.eval()

    return model


# ============================================================
# Main benchmark
# ============================================================

@torch.no_grad()
def main():

    print()
    print("=" * 70)
    print("REACT-EEG Benchmark")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    if DEVICE == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Patient:",
        PATIENT
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = build_dataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_labels = []

    all_snn_logits = []

    all_transformer_logits = []

    all_final_logits = []

    all_trigger = []

    # --------------------------------------------------------
    # Execution statistics
    # --------------------------------------------------------

    total_samples = 0

    total_trigger = 0

    total_transformer_execution = 0

    transformer_call_count = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Running benchmark")
    print("=" * 70)

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Reset SNN membrane states
        # ----------------------------------------------------

        reset_network(model)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        out = model(x)

        snn_logits = out["snn_logits"]

        transformer_logits = out["transformer_logits"]

        final_logits = out["logits"]

        trigger = out["trigger"]

        batch_size = x.size(0)

        total_samples += batch_size

        # ----------------------------------------------------
        # Trigger statistics
        # ----------------------------------------------------

        trigger_count = (
            trigger.float()
            .sum()
            .item()
        )

        total_trigger += trigger_count

        # ----------------------------------------------------
        # Transformer execution
        #
        # In our current model:
        #
        # trigger=True
        #
        # means Transformer is executed.
        # ----------------------------------------------------

        executed_count = trigger_count

        total_transformer_execution += (
            executed_count
        )

        if executed_count > 0:

            transformer_call_count += 1

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        all_labels.append(
            y.detach().cpu()
        )

        all_snn_logits.append(
            snn_logits.detach().cpu()
        )

        all_transformer_logits.append(
            transformer_logits.detach().cpu()
        )

        all_final_logits.append(
            final_logits.detach().cpu()
        )

        all_trigger.append(
            trigger.detach().cpu()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx % 100 == 0
            or batch_idx == len(loader) - 1
        ):

            current_trigger_rate = (
                total_trigger
                / total_samples
            )

            current_execution_rate = (
                total_transformer_execution
                / total_samples
            )

            print(
                f"Batch "
                f"{batch_idx + 1:5d}/"
                f"{len(loader):5d} | "
                f"Trigger: "
                f"{current_trigger_rate * 100:6.2f}% | "
                f"Transformer execution: "
                f"{current_execution_rate * 100:6.2f}%"
            )

    # ========================================================
    # Concatenate
    # ========================================================

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    all_snn_logits = torch.cat(
        all_snn_logits,
        dim=0
    )

    all_transformer_logits = torch.cat(
        all_transformer_logits,
        dim=0
    )

    all_final_logits = torch.cat(
        all_final_logits,
        dim=0
    )

    all_trigger = torch.cat(
        all_trigger,
        dim=0
    )

    # ========================================================
    # Metrics
    # ========================================================

    snn_metrics = calculate_metrics(
        all_snn_logits,
        all_labels
    )

    transformer_metrics = calculate_metrics(
        all_transformer_logits,
        all_labels
    )

    final_metrics = calculate_metrics(
        all_final_logits,
        all_labels
    )

    # ========================================================
    # Print comparison
    # ========================================================

    print()
    print("=" * 70)
    print("SNN vs Transformer vs REACT-EEG")
    print("=" * 70)

    print_metrics(
        "SNN ONLY",
        snn_metrics
    )

    print_metrics(
        "TRANSFORMER BRANCH",
        transformer_metrics
    )

    print_metrics(
        "FINAL REACT-EEG",
        final_metrics
    )

    # ========================================================
    # Routing statistics
    # ========================================================

    trigger_rate = (
        total_trigger
        / total_samples
        if total_samples > 0
        else 0.0
    )

    execution_rate = (
        total_transformer_execution
        / total_samples
        if total_samples > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("Dynamic Routing Statistics")
    print("=" * 70)

    print(
        f"Total samples: "
        f"{total_samples}"
    )

    print(
        f"Triggered samples: "
        f"{int(total_trigger)}"
    )

    print(
        f"Trigger rate: "
        f"{trigger_rate * 100:.2f}%"
    )

    print(
        f"Transformer executed samples: "
        f"{int(total_transformer_execution)}"
    )

    print(
        f"Transformer execution rate: "
        f"{execution_rate * 100:.2f}%"
    )

    print(
        f"Transformer call count: "
        f"{transformer_call_count}"
    )

    # ========================================================
    # Error relationship
    # ========================================================

    snn_pred = torch.argmax(
        all_snn_logits,
        dim=1
    )

    transformer_pred = torch.argmax(
        all_transformer_logits,
        dim=1
    )

    snn_correct = (
        snn_pred == all_labels
    )

    transformer_correct = (
        transformer_pred == all_labels
    )

    snn_wrong_transformer_correct = (
        (~snn_correct)
        & transformer_correct
    ).sum().item()

    snn_correct_transformer_wrong = (
        snn_correct
        & (~transformer_correct)
    ).sum().item()

    both_wrong = (
        (~snn_correct)
        & (~transformer_correct)
    ).sum().item()

    both_correct = (
        snn_correct
        & transformer_correct
    ).sum().item()

    print()
    print("=" * 70)
    print("Error Relationship")
    print("=" * 70)

    print(
        "SNN wrong -> Transformer correct:"
    )

    print(
        f"    {snn_wrong_transformer_correct}"
    )

    print(
        "SNN correct -> Transformer wrong:"
    )

    print(
        f"    {snn_correct_transformer_wrong}"
    )

    print(
        "Both wrong:"
    )

    print(
        f"    {both_wrong}"
    )

    print(
        "Both correct:"
    )

    print(
        f"    {both_correct}"
    )

    # ========================================================
    # Seizure-specific analysis
    # ========================================================

    seizure_mask = (
        all_labels == 1
    )

    seizure_count = (
        seizure_mask.sum().item()
    )

    seizure_snn_wrong = (
        seizure_mask
        & (~snn_correct)
    )

    seizure_snn_wrong_transformer_correct = (
        seizure_snn_wrong
        & transformer_correct
    )

    seizure_both_wrong = (
        seizure_mask
        & (~snn_correct)
        & (~transformer_correct)
    )

    seizure_transformer_correct = (
        seizure_mask
        & transformer_correct
    )

    print()
    print("=" * 70)
    print("Seizure-specific Error Analysis")
    print("=" * 70)

    print(
        "Total seizure samples:"
    )

    print(
        f"    {seizure_count}"
    )

    print(
        "Seizure: SNN wrong -> Transformer correct:"
    )

    print(
        f"    "
        f"{seizure_snn_wrong_transformer_correct.sum().item()}"
    )

    print(
        "Seizure: SNN wrong -> Transformer wrong:"
    )

    print(
        f"    "
        f"{seizure_both_wrong.sum().item()}"
    )

    print(
        "Seizure: Transformer correct:"
    )

    print(
        f"    "
        f"{seizure_transformer_correct.sum().item()}"
    )

    # ========================================================
    # Interpretation
    # ========================================================

    print()
    print("=" * 70)
    print("Benchmark Interpretation")
    print("=" * 70)

    if (
        snn_wrong_transformer_correct
        > 0
    ):

        print(
            "Transformer can correct some SNN errors."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            "Transformer did not correct any SNN errors."
        )

    if (
        seizure_snn_wrong_transformer_correct.sum().item()
        > 0
    ):

        print(
            "Transformer corrected at least "
            "one seizure-specific SNN error."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            "Transformer did not correct "
            "any seizure-specific SNN errors."
        )

    if execution_rate > 0.8:

        print()
        print(
            "WARNING:"
        )

        print(
            "Transformer execution rate is above 80%."
        )

        print(
            "The current routing policy is not "
            "saving much computation."
        )

    elif execution_rate < 0.3:

        print()
        print(
            "Transformer execution rate is below 30%."
        )

        print(
            "The current router is relatively selective."
        )

    else:

        print()
        print(
            "Transformer execution rate is between "
            "30% and 80%."
        )

    print()
    print("=" * 70)
    print("Benchmark finished")
    print("=" * 70)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()