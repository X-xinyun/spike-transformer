import torch
import torch.nn as nn

from spikingjelly.clock_driven import neuron
from spikingjelly.clock_driven import functional

class SNNBlock(nn.Module):
    """
    Basic 1D Spiking Neural Network block.

    Structure:
        Conv1d -> BatchNorm1d -> LIF
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        padding=3
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_channels)

        self.lif = neuron.MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
            backend='torch'
        )

    def forward(self, x):
        functional.reset_net(self)
             # --------------------------------------------------
        # Reset SNN neuron states
        # --------------------------------------------------
        #
        # MultiStepLIFNode contains membrane states.
        # We must reset them before processing a new EEG batch.
        #
        # Otherwise the computation graph from the previous
        # batch can remain connected to the current batch,
        # causing:
        #
        # "Trying to backward through the graph a second time"
        #
        # --------------------------------------------------

       
        """
        x:
            [T, B, C, L]

        return:
            [T, B, C_out, L_out]
        """

        T, B, C, L = x.shape

        # Merge time and batch temporarily
        x = x.reshape(T * B, C, L)

        x = self.conv(x)
        x = self.bn(x)

        # Restore time dimension
        _, C_out, L_out = x.shape
        x = x.reshape(T, B, C_out, L_out)

        # LIF neuron
        x = self.lif(x)

        return x


class SNNSentinel(nn.Module):
    """
    REACT-EEG Stage-1 Spiking Sentinel.

    Input:
        EEG tensor [B, C, L]

        B = batch size
        C = EEG channels
        L = temporal samples

    Example:
        [2, 22, 1280]

    Outputs:
        risk:
            [B]

        uncertainty:
            [B]

        channel_reliability:
            [B, C]
    """

    def __init__(
        self,
        in_channels=22,
        time_steps=4,
        hidden_channels=32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.hidden_channels = hidden_channels

        # --------------------------------------------------
        # SNN feature extractor
        # --------------------------------------------------

        self.block1 = SNNBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.block2 = SNNBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=7,
            stride=2,
            padding=3
        )

        feature_channels = hidden_channels * 2

        # --------------------------------------------------
        # Global feature pooling
        # --------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --------------------------------------------------
        # Stage-1 prediction heads
        # --------------------------------------------------

        # Seizure risk
        self.risk_head = nn.Sequential(
            nn.Linear(feature_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Prediction uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Channel reliability
        self.channel_head = nn.Sequential(
            nn.Linear(feature_channels, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            EEG input.

            Shape:
                [B, C, L]

            Example:
                [2, 22, 1280]

        Returns
        -------
        dict
            Dictionary containing:

            risk
                [B]

            uncertainty
                [B]

            channel_reliability
                [B, C]

            feature
                [B, feature_channels]
        """

        # --------------------------------------------------
        # Check input
        # --------------------------------------------------

        if x.dim() != 3:
            raise ValueError(
                "SNNSentinel expects input shape "
                "[B, C, L], but received "
                f"{tuple(x.shape)}"
            )

        B, C, L = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} EEG channels, "
                f"but received {C}"
            )

        # --------------------------------------------------
        # Create temporal copies for SNN
        # --------------------------------------------------
        #
        # Original EEG:
        #
        # [B, C, L]
        #
        # SNN:
        #
        # [T, B, C, L]
        #
        # where T = time_steps
        # --------------------------------------------------

        x = x.unsqueeze(0).repeat(
            self.time_steps,
            1,
            1,
            1
        )

        # --------------------------------------------------
        # SNN feature extraction
        # --------------------------------------------------

        x = self.block1(x)

        x = self.block2(x)

        # --------------------------------------------------
        # Average spike activity over time
        # --------------------------------------------------

        x = x.mean(dim=0)

        # x:
        # [B, feature_channels, L]
        # --------------------------------------------------

        x = self.global_pool(x)

        # [B, feature_channels, 1]
        x = x.squeeze(-1)

        # [B, feature_channels]
        feature = x

        # --------------------------------------------------
        # Risk
        # --------------------------------------------------

        risk = torch.sigmoid(
            self.risk_head(feature)
        ).squeeze(-1)

        # --------------------------------------------------
        # Uncertainty
        # --------------------------------------------------

        uncertainty = torch.sigmoid(
            self.uncertainty_head(feature)
        ).squeeze(-1)

        # --------------------------------------------------
        # Channel reliability
        # --------------------------------------------------

        channel_reliability = torch.sigmoid(
            self.channel_head(feature)
        )

        # --------------------------------------------------
        # Return evidence information
        # --------------------------------------------------

        return {
            "risk": risk,
            "uncertainty": uncertainty,
            "channel_reliability": channel_reliability,
            "feature": feature
        }