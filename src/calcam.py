"""
Generic implementation of CALCAM.

CALCAM selects the most aligned modality pair using CKA, applies
bidirectional cross-attention to this pair, independently processes
the remaining modality through self-attention, and integrates the
three representations through a shared Transformer.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CALCAM(nn.Module):
    """
    CKA-guided multimodal fusion architecture for three modalities.

    Parameters
    ----------
    input_dims:
        Feature dimensions of the three input modalities.

    selected_pair:
        Indices of the two modalities selected using pairwise CKA.
        For example, ``(1, 2)`` selects modalities M2 and M3.

    d_model:
        Common Transformer embedding dimension.

    nhead:
        Number of attention heads.

    num_shared_layers:
        Number of layers in the shared Transformer stage.

    dim_feedforward:
        Hidden dimension of Transformer feed-forward networks.

    dropout:
        Dropout probability.

    num_classes:
        Number of output classes. Use ``1`` for binary classification
        with ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        selected_pair: tuple[int, int],
        d_model: int = 256,
        nhead: int = 8,
        num_shared_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()

        if len(input_dims) != 3:
            raise ValueError(
                "CALCAM currently expects exactly three modalities."
            )

        if len(selected_pair) != 2:
            raise ValueError(
                "selected_pair must contain exactly two modality indices."
            )

        first, second = selected_pair

        if first == second:
            raise ValueError(
                "The two selected modalities must be different."
            )

        if first not in range(3) or second not in range(3):
            raise ValueError(
                "Modality indices must belong to {0, 1, 2}."
            )

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead."
            )

        self.input_dims = tuple(input_dims)
        self.selected_pair = tuple(selected_pair)
        self.remaining_index = next(
            index
            for index in range(3)
            if index not in selected_pair
        )

        self.d_model = d_model
        self.num_classes = num_classes

        # Modality-specific projections into a common space.
        self.projections = nn.ModuleList(
            [
                nn.Linear(input_dim, d_model)
                for input_dim in input_dims
            ]
        )

        # A <- B
        self.cross_attention_ab = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # B <- A
        self.cross_attention_ba = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Independent self-attention for modality C.
        self.self_attention_c = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ab = nn.LayerNorm(d_model)
        self.norm_ba = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)

        self.ffn_ab = self._build_feedforward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.ffn_ba = self._build_feedforward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.ffn_c = self._build_feedforward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.ffn_norm_ab = nn.LayerNorm(d_model)
        self.ffn_norm_ba = nn.LayerNorm(d_model)
        self.ffn_norm_c = nn.LayerNorm(d_model)

        # Shared Transformer after concatenating the three sequences.
        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.shared_transformer = nn.TransformerEncoder(
            shared_layer,
            num_layers=num_shared_layers,
        )

        self.final_norm = nn.LayerNorm(d_model)

        output_dim = 1 if num_classes == 1 else num_classes

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    @staticmethod
    def _build_feedforward(
        d_model: int,
        dim_feedforward: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _ensure_token_sequence(
        representation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a pooled representation (B, D) into one token (B, 1, D).

        Token-level inputs with shape (B, N, D) are preserved.
        """
        if representation.ndim == 2:
            return representation.unsqueeze(1)

        if representation.ndim == 3:
            return representation

        raise ValueError(
            "Each modality must have shape (batch, features) or "
            "(batch, tokens, features). "
            f"Received {tuple(representation.shape)}."
        )

    def _project_modalities(
        self,
        modalities: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(modalities) != 3:
            raise ValueError(
                "Exactly three modality tensors must be supplied."
            )

        projected = []

        for index, (representation, projection) in enumerate(
            zip(modalities, self.projections)
        ):
            representation = self._ensure_token_sequence(
                representation
            )

            if representation.shape[-1] != self.input_dims[index]:
                raise ValueError(
                    f"Modality {index} has feature dimension "
                    f"{representation.shape[-1]}, but "
                    f"{self.input_dims[index]} was expected."
                )

            projected.append(
                projection(representation)
            )

        batch_sizes = {
            representation.shape[0]
            for representation in projected
        }

        if len(batch_sizes) != 1:
            raise ValueError(
                "All modalities must have the same batch size."
            )

        return projected

    def forward(
        self,
        modalities: Sequence[torch.Tensor],
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        modalities:
            Sequence containing three modality tensors.

            Each tensor may have shape:

            - ``(batch, features)`` for pooled embeddings;
            - ``(batch, tokens, features)`` for token-level embeddings.

        return_attention:
            If True, also return cross-attention and self-attention weights.

        Returns
        -------
        logits:
            Shape ``(batch,)`` for binary classification or
            ``(batch, num_classes)`` for multiclass classification.
        """

        projected = self._project_modalities(modalities)

        index_a, index_b = self.selected_pair
        index_c = self.remaining_index

        modality_a = projected[index_a]
        modality_b = projected[index_b]
        modality_c = projected[index_c]

        # A <- B
        attended_ab, weights_ab = self.cross_attention_ab(
            query=modality_a,
            key=modality_b,
            value=modality_b,
            need_weights=return_attention,
        )

        attended_ab = self.norm_ab(
            modality_a + attended_ab
        )

        attended_ab = self.ffn_norm_ab(
            attended_ab + self.ffn_ab(attended_ab)
        )

        # B <- A
        attended_ba, weights_ba = self.cross_attention_ba(
            query=modality_b,
            key=modality_a,
            value=modality_a,
            need_weights=return_attention,
        )

        attended_ba = self.norm_ba(
            modality_b + attended_ba
        )

        attended_ba = self.ffn_norm_ba(
            attended_ba + self.ffn_ba(attended_ba)
        )

        # Independent self-attention for C.
        attended_c, weights_c = self.self_attention_c(
            query=modality_c,
            key=modality_c,
            value=modality_c,
            need_weights=return_attention,
        )

        attended_c = self.norm_c(
            modality_c + attended_c
        )

        attended_c = self.ffn_norm_c(
            attended_c + self.ffn_c(attended_c)
        )

        # Concatenate token sequences and apply a shared Transformer.
        fused_tokens = torch.cat(
            [
                attended_ab,
                attended_ba,
                attended_c,
            ],
            dim=1,
        )

        fused_tokens = self.shared_transformer(
            fused_tokens
        )

        fused_tokens = self.final_norm(
            fused_tokens
        )

        # Global mean pooling over the concatenated token sequence.
        fused_representation = fused_tokens.mean(dim=1)

        logits = self.classifier(
            fused_representation
        )

        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_attention:
            attention = {
                "A_from_B": weights_ab,
                "B_from_A": weights_ba,
                "C_self": weights_c,
            }

            return logits, attention

        return logits
