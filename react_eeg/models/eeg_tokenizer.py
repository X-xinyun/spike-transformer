import torch
import torch.nn as nn


class EEGTokenizer(nn.Module):
    """
    EEG Tokenizer

    Input:
        x: [B, C, T]

        B = batch size
        C = EEG channels
        T = temporal samples

    Output:
        tokens: [B, N, D]

        N = number of EEG tokens
        D = embedding dimension

    Example:
        Input:
            [B, 8, 1280]

        With:
            patch_size = 64
            embed_dim = 128

        Output:
            [B, 20, 128]
    """

    def __init__(
        self,
        in_channels=8,
        patch_size=64,
        embed_dim=128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # ---------------------------------------------------------
        # Temporal patch embedding
        #
        # Conv1d:
        #
        # [B, C, T]
        #       ↓
        # [B, D, N]
        #
        # kernel_size = patch_size
        # stride      = patch_size
        #
        # This divides the EEG signal into non-overlapping
        # temporal patches.
        # ---------------------------------------------------------
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Normalize each token embedding.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x:
                EEG tensor with shape [B, C, T]

        Returns:
            tokens:
                Tensor with shape [B, N, D]
        """

        # Check input dimension
        if x.dim() != 3:
            raise ValueError(
                "EEGTokenizer expects input with shape [B, C, T], "
                f"but got {tuple(x.shape)}"
            )

        # Check channel number
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} EEG channels, "
                f"but got {x.size(1)}"
            )

        # ---------------------------------------------------------
        # Temporal patch embedding
        #
        # [B, C, T]
        #      ↓ Conv1D
        # [B, D, N]
        # ---------------------------------------------------------
        x = self.proj(x)

        # ---------------------------------------------------------
        # Transformer expects:
        #
        # [B, N, D]
        #
        # so transpose channel/embedding dimension and token
        # dimension.
        # ---------------------------------------------------------
        x = x.transpose(1, 2)

        # ---------------------------------------------------------
        # Layer normalization
        # ---------------------------------------------------------
        x = self.norm(x)

        return x


if __name__ == "__main__":
    # -------------------------------------------------------------
    # Simple standalone test
    # -------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("EEG Tokenizer Test")
    print("=" * 70)

    # Example:
    # 8 selected EEG channels
    # 5 seconds
    # 256 Hz
    #
    # 5 * 256 = 1280 samples
    x = torch.randn(
        2,
        8,
        1280,
        device=device,
    )

    model = EEGTokenizer(
        in_channels=8,
        patch_size=64,
        embed_dim=128,
    ).to(device)

    tokens = model(x)

    print("Input shape:")
    print(x.shape)

    print()

    print("Output shape:")
    print(tokens.shape)

    print()

    print("Expected:")
    print("[2, 20, 128]")

    print()

    # Test backward
    loss = tokens.mean()
    loss.backward()

    print("Backward:")
    print("OK")

    print("=" * 70)
    print("EEG Tokenizer test finished")
    print("=" * 70)