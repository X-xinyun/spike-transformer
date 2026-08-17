import torch
import torch.nn as nn


class EvidenceRouter(nn.Module):
    """
    REACT-EEG Evidence Router

    Stage:
        SNN Sentinel
              |
              +--> risk
              +--> uncertainty
              +--> channel reliability
              |
              v
        Evidence Router
              |
              +--> evidence
              +--> trigger

    Input:
        risk:
            [B]

        uncertainty:
            [B]

        channel_reliability:
            [B, C]

    Output:
        evidence:
            [B]

        trigger:
            [B]

        mean_reliability:
            [B]
    """

    def __init__(
        self,
        risk_weight=0.6,
        uncertainty_weight=0.2,
        reliability_weight=0.2,
        trigger_threshold=0.5,
    ):
        super().__init__()

        # --------------------------------------------------
        # Evidence weights
        # --------------------------------------------------

        self.risk_weight = risk_weight

        self.uncertainty_weight = uncertainty_weight

        self.reliability_weight = reliability_weight

        # --------------------------------------------------
        # Trigger threshold
        # --------------------------------------------------

        self.trigger_threshold = trigger_threshold

    def forward(
        self,
        risk,
        uncertainty,
        channel_reliability,
    ):

        # ==================================================
        # 1. Check input
        # ==================================================

        if risk.dim() != 1:
            raise ValueError(
                "risk must have shape [B], "
                f"but received {tuple(risk.shape)}"
            )

        if uncertainty.dim() != 1:
            raise ValueError(
                "uncertainty must have shape [B], "
                f"but received {tuple(uncertainty.shape)}"
            )

        if channel_reliability.dim() != 2:
            raise ValueError(
                "channel_reliability must have shape [B, C], "
                f"but received "
                f"{tuple(channel_reliability.shape)}"
            )

        # ==================================================
        # 2. Mean channel reliability
        # ==================================================

        mean_reliability = (
            channel_reliability.mean(dim=1)
        )

        # ==================================================
        # 3. Convert reliability into unreliability
        #
        # High reliability:
        #
        #     reliability -> 1
        #
        # means:
        #
        #     unreliability -> 0
        #
        # Low reliability:
        #
        #     reliability -> 0
        #
        # means:
        #
        #     unreliability -> 1
        # ==================================================

        unreliability = (
            1.0 - mean_reliability
        )

        # ==================================================
        # 4. Evidence
        #
        # Evidence consists of three parts:
        #
        #     seizure risk
        #     +
        #     uncertainty
        #     +
        #     channel unreliability
        #
        # Higher evidence means:
        #
        #     "This EEG deserves more computation."
        # ==================================================

        evidence = (
            self.risk_weight * risk
            +
            self.uncertainty_weight * uncertainty
            +
            self.reliability_weight * unreliability
        )

        # ==================================================
        # 5. Clamp evidence
        #
        # Since all three inputs are expected to be in [0,1],
        # evidence should also remain in [0,1].
        # ==================================================

        evidence = torch.clamp(
            evidence,
            min=0.0,
            max=1.0
        )

        # ==================================================
        # 6. Trigger Transformer
        #
        # IMPORTANT:
        #
        # This is currently a hard routing decision.
        #
        # Training loss should NOT backpropagate through
        # this boolean tensor.
        #
        # The differentiable evidence itself is used by
        # the final fusion.
        # ==================================================

        trigger = (
            evidence >= self.trigger_threshold
        )

        return {
            "trigger": trigger,

            "evidence": evidence,

            "mean_reliability": mean_reliability,

            "unreliability": unreliability,
        }


if __name__ == "__main__":

    print("=" * 70)
    print("REACT-EEG Evidence Router Test")
    print("=" * 70)

    # --------------------------------------------------
    # Fake input
    # --------------------------------------------------

    B = 4

    C = 22

    risk = torch.tensor(
        [
            0.1,
            0.8,
            0.2,
            0.9
        ]
    )

    uncertainty = torch.tensor(
        [
            0.1,
            0.2,
            0.8,
            0.1
        ]
    )

    reliability = torch.tensor(
        [
            [0.9] * C,
            [0.9] * C,
            [0.9] * C,
            [0.1] * C,
        ]
    )

    # --------------------------------------------------
    # Router
    # --------------------------------------------------

    router = EvidenceRouter()

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    output = router(
        risk=risk,
        uncertainty=uncertainty,
        channel_reliability=reliability
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print()
    print("Risk:")
    print(risk)

    print()
    print("Uncertainty:")
    print(uncertainty)

    print()
    print("Mean reliability:")
    print(output["mean_reliability"])

    print()
    print("Unreliability:")
    print(output["unreliability"])

    print()
    print("Evidence:")
    print(output["evidence"])

    print()
    print("Trigger:")
    print(output["trigger"])

    print()
    print("=" * 70)
    print("Evidence Router test finished")
    print("=" * 70)