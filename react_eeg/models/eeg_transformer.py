import torch
import torch.nn as nn

from .eeg_tokenizer import EEGTokenizer


class EEGTransformer(nn.Module):
    """
    First-version EEG Transformer for REACT-EEG.

    Input:
        x: [B, C, T]

        B = batch size
        C = selected EEG channels
        T = temporal samples

    Output:
        logits: [B, num_classes]

    Current design:

        EEG
          |
          v
        EEGTokenizer
          |
          v
        [B, N, D]
          |
          v
        Transformer Encoder
          |
          v
        Mean Pooling
          |
          v
        Classifier
    """

    def __init__(
        self,
        in_channels=8,
        patch_size=64,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ---------------------------------------------------------
        # 1. EEG Tokenizer
        #
        # [B, C, T]
        #      ↓
        # [B, N, D]
        # ---------------------------------------------------------
        self.tokenizer = EEGTokenizer(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # ---------------------------------------------------------
        # 2. Learnable positional embedding
        #
        # For our current CHB-MIT setting:
        #
        # T = 1280
        # patch_size = 64
        #
        # N = 1280 / 64 = 20
        #
        # Therefore:
        #
        # [1, 20, 128]
        # ---------------------------------------------------------
        num_tokens = 1280 // patch_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )

        # ---------------------------------------------------------
        # 3. Transformer Encoder
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---------------------------------------------------------
        # 4. Final normalization
        # ---------------------------------------------------------
        self.norm = nn.LayerNorm(embed_dim)

        # ---------------------------------------------------------
        # 5. Classification head
        #
        # Binary seizure classification:
        #
        # 0 = non-seizure
        # 1 = seizure
        # ---------------------------------------------------------
        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

        # Initialize positional embedding
        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

    def forward(self, x):
        """
        Args:
            x:
                [B, C, T]

        Returns:
            logits:
                [B, num_classes]
        """

        # ---------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                "EEGTransformer expects [B, C, T], "
                f"but got {tuple(x.shape)}"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"but got {x.size(1)}"
            )

        # ---------------------------------------------------------
        # Tokenization
        #
        # [B, C, T]
        #      ↓
        # [B, N, D]
        # ---------------------------------------------------------
        x = self.tokenizer(x)

        # ---------------------------------------------------------
        # Positional embedding
        # ---------------------------------------------------------
        if x.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"Expected {self.pos_embed.size(1)} tokens, "
                f"but got {x.size(1)}"
            )

        x = x + self.pos_embed

        # ---------------------------------------------------------
        # Transformer
        #
        # [B, N, D]
        #      ↓
        # [B, N, D]
        # ---------------------------------------------------------
        x = self.encoder(x)

        # ---------------------------------------------------------
        # Mean pooling over temporal tokens
        #
        # [B, N, D]
        #      ↓
        # [B, D]
        # ---------------------------------------------------------
        x = x.mean(dim=1)

        # ---------------------------------------------------------
        # Normalization
        # ---------------------------------------------------------
        x = self.norm(x)

        # ---------------------------------------------------------
        # Classification
        #
        # [B, D]
        #      ↓
        # [B, 2]
        # ---------------------------------------------------------
        logits = self.head(x)

        return logits


if __name__ == "__main__":

    print("=" * 70)
    print("EEG Transformer Test")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------
    # Simulated selected EEG data
    #
    # 8 channels
    # 5 seconds
    # 256 Hz
    #
    # [B, 8, 1280]
    # -------------------------------------------------------------
    x = torch.randn(
        2,
        8,
        1280,
        device=device,
    )

    # -------------------------------------------------------------
    # Create Transformer
    # -------------------------------------------------------------
    model = EEGTransformer(
        in_channels=8,
        patch_size=64,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        num_classes=2,
    ).to(device)

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------
    logits = model(x)

    print("Input:")
    print(x.shape)

    print()

    print("Logits:")
    print(logits.shape)

    print()

    print("Expected:")
    print("[2, 2]")

    # -------------------------------------------------------------
    # Fake labels
    # -------------------------------------------------------------
    target = torch.tensor(
        [0, 1],
        device=device,
        dtype=torch.long,
    )

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        target,
    )

    print()

    print("Loss:")
    print(loss.item())

    # -------------------------------------------------------------
    # Backward
    # -------------------------------------------------------------
    loss.backward()

    print()

    print("Backward:")
    print("OK")

    print("=" * 70)
    print("EEG Transformer test finished")
    print("=" * 70)