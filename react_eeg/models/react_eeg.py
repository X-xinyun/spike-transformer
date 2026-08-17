import torch
import torch.nn as nn

from .snn_sentinel import SNNSentinel
from .eeg_transformer import EEGTransformer
from .react_router import EvidenceRouter


class REACTEEG(nn.Module):
    """
    First-version REACT-EEG

    Training:
        EEG
          |
          v
    SNN Sentinel
          |
          +--> risk
          +--> uncertainty
          +--> channel reliability
          +--> feature
          |
          v
    Evidence Router
          |
          +--> trigger
          +--> evidence
          |
          +--------------------+
          |                    |
          v                    v
      SNN Classifier      EEG Transformer
          |                    |
          +---------+----------+
                    |
                    v
              Evidence Fusion
                    |
                    v
              Final logits


    Evaluation / Inference:

        EEG
          |
          v
    SNN Sentinel
          |
          v
    Evidence Router
          |
          v
        Trigger
        /     \
     False    True
       |        |
       v        v
      SNN    Transformer
       |        |
       +----+---+
            |
            v
       Final logits
    """

    def __init__(
        self,
        in_channels=22,
        time_steps=4,
        snn_hidden_channels=32,
        transformer_channels=8,
        num_classes=2,
        embed_dim=128,
        num_heads=8,
        num_layers=2,
    ):
        super().__init__()

        # ============================================================
        # Stage 1
        # Always-on SNN Sentinel
        # ============================================================

        self.snn_sentinel = SNNSentinel(
            in_channels=in_channels,
            time_steps=time_steps,
            hidden_channels=snn_hidden_channels
        )

        # ============================================================
        # Stage 2
        # EEG Transformer
        #
        # First version:
        # use the first 8 EEG channels.
        #
        # Later:
        # replace with reliability-based channel selection.
        # ============================================================

        self.transformer = EEGTransformer(
            in_channels=transformer_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # ============================================================
        # Stage 3
        # Evidence Router
        # ============================================================

        self.router = EvidenceRouter()

        # ============================================================
        # SNN classifier
        #
        # SNNSentinel feature dimension = 64
        # ============================================================

        self.snn_classifier = nn.Linear(
            64,
            num_classes
        )

        self.transformer_channels = transformer_channels

    def forward(self, x):

        # ============================================================
        # Input
        #
        # x:
        # [B, 22, 1280]
        # ============================================================

        batch_size = x.size(0)

        # ============================================================
        # 1. Always-on SNN Sentinel
        # ============================================================

        snn_out = self.snn_sentinel(x)

        risk = snn_out["risk"]

        uncertainty = snn_out["uncertainty"]

        channel_reliability = snn_out["channel_reliability"]

        snn_feature = snn_out["feature"]

        # ============================================================
        # 2. Evidence Router
        # ============================================================

        router_out = self.router(
            risk=risk,
            uncertainty=uncertainty,
            channel_reliability=channel_reliability
        )

        trigger = router_out["trigger"]

        evidence = router_out["evidence"]
        mean_reliability = router_out["mean_reliability"]

        # ============================================================
        # 3. SNN prediction
        # ============================================================

        snn_logits = self.snn_classifier(
            snn_feature
        )

        # ============================================================
        # 4. Transformer
        #
        # IMPORTANT:
        #
        # During training:
        #     Transformer processes the whole batch.
        #
        # During evaluation:
        #     Transformer processes ONLY trigger=True samples.
        #
        # This is our first implementation of
        # adaptive computation.
        # ============================================================

        transformer_logits = torch.zeros(
            batch_size,
            snn_logits.size(1),
            device=x.device,
            dtype=snn_logits.dtype
        )

        # ------------------------------------------------------------
        # Training
        # ------------------------------------------------------------

        if self.training:

            # ------------------------------------------------------------
            # 4. Reliability-guided Top-K channel selection
            #
            #  channel_reliability:
            # [B, 22]
            #
            # Select the 8 most reliable EEG channels for Transformer.
            # ------------------------------------------------------------

            top_k = 8

            _, topk_indices = torch.topk(
                channel_reliability,
                k=top_k,
                dim=1
            )

    # [B, 22] -> [B, 8]
    # Expand indices along temporal dimension.
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(
              -1,
              -1,
              x.size(-1)
            )

    # [B, 22, 1280] -> [B, 8, 1280]
            transformer_x = torch.gather(
                  x,
                  dim=1,
                  index=topk_indices_expanded
            )

    # Transformer
            transformer_logits = self.transformer(
            transformer_x
            )
            

        # ------------------------------------------------------------
        # Evaluation / inference
        # ------------------------------------------------------------

        else:

            if trigger.any():

                trigger_indices = torch.nonzero(
                    trigger,
                    as_tuple=False
                ).squeeze(1)

                transformer_x = x[
                    trigger_indices,
                    :self.transformer_channels,
                    :
                ]

                selected_transformer_logits = self.transformer(
                    transformer_x
                )

                transformer_logits[
                    trigger_indices
                ] = selected_transformer_logits

        # ============================================================
        # 5. Evidence Fusion
        #
        # IMPORTANT:
        #
        # For triggered samples:
        #
        #   final =
        #       (1-evidence) * SNN
        #       +
        #       evidence * Transformer
        #
        # For non-triggered samples:
        #
        #   final = SNN
        #
        # This guarantees that Transformer is not required
        # for non-triggered inference.
        # ============================================================

        evidence_weight = evidence.unsqueeze(1)

        if self.training:

            final_logits = (
                (1.0 - evidence_weight) * snn_logits
                +
                evidence_weight * transformer_logits
            )

        else:

            final_logits = snn_logits.clone()

            if trigger.any():

                trigger_indices = torch.nonzero(
                    trigger,
                    as_tuple=False
                ).squeeze(1)

                selected_snn_logits = snn_logits[
                    trigger_indices
                ]

                selected_transformer_logits = transformer_logits[
                    trigger_indices
                ]

                selected_evidence = evidence_weight[
                    trigger_indices
                ]

                selected_final_logits = (
                    (1.0 - selected_evidence)
                    * selected_snn_logits
                    +
                    selected_evidence
                    * selected_transformer_logits
                )

                final_logits[
                    trigger_indices
                ] = selected_final_logits

        # ============================================================
        # 6. Return all intermediate results
        # ============================================================

        return {
            "logits": final_logits,

            "snn_logits": snn_logits,

            "transformer_logits": transformer_logits,

            "risk": risk,

            "uncertainty": uncertainty,

            "channel_reliability": channel_reliability,

            "evidence": evidence,
            "mean_reliability": mean_reliability,


            "trigger": trigger,

            "snn_feature": snn_feature,
        }


# ====================================================================
# Test
# ====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("REACT-EEG Dynamic Routing Test")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    # ================================================================
    # Model
    # ================================================================

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

    # ================================================================
    # Fake EEG
    # ================================================================

    x = torch.randn(
        4,
        22,
        1280,
        device=device
    )

    y = torch.tensor(
        [0, 1, 0, 1],
        device=device
    )

    # ================================================================
    # Training test
    # ================================================================

    print()
    print("-" * 70)
    print("Training mode")
    print("-" * 70)

    model.train()

    out = model(x)

    print("Input:")
    print(x.shape)

    print()
    print("Final logits:")
    print(out["logits"].shape)

    print()
    print("SNN logits:")
    print(out["snn_logits"].shape)

    print()
    print("Transformer logits:")
    print(out["transformer_logits"].shape)

    print()
    print("Risk:")
    print(out["risk"])

    print()
    print("Uncertainty:")
    print(out["uncertainty"])

    print()
    print("Evidence:")
    print(out["evidence"])

    print()
    print("Mean reliability:")
    print(out["mean_reliability"])

    print()
    print("Trigger:")
    print(out["trigger"])

    # ================================================================
    # Loss
    # ================================================================

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        out["logits"],
        y
    )

    print()
    print("Training loss:")
    print(loss.item())

    # ================================================================
    # Backward
    # ================================================================

    loss.backward()

    print()
    print("Training backward:")
    print("OK")

    # ================================================================
    # Evaluation test
    # ================================================================

    print()
    print("-" * 70)
    print("Evaluation mode")
    print("-" * 70)

    model.eval()

    with torch.no_grad():

        out_eval = model(x)

    print()
    print("Evaluation logits:")
    print(out_eval["logits"].shape)

    print()
    print("Evaluation trigger:")
    print(out_eval["trigger"])

    trigger_rate = out_eval["trigger"].float().mean().item()

    print()
    print("Transformer trigger rate:")
    print("{:.2f}%".format(trigger_rate * 100))

    print()
    print("Evaluation test:")
    print("OK")

    print("=" * 70)
    print("REACT-EEG dynamic routing test finished")
    print("=" * 70)
