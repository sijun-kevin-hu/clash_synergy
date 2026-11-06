"""Independent-deck baseline"""
import torch
from torch import nn


class IndependentDeckNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=32,
        encoder_hidden=64,
        encoder_out=64,
        mlp_hidden=64,
        player_stat_dim=0,
        dropout=0.1,
    ):
        super().__init__()
        if min(vocab_size, embed_dim, encoder_hidden, encoder_out, mlp_hidden) < 1:
            raise ValueError("Architecture sizes must be positive.")
        if player_stat_dim < 0 or not 0 <= dropout < 1:
            raise ValueError("player_stat_dim must be non-negative and dropout must be in [0, 1).")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_hidden = encoder_hidden
        self.encoder_out = encoder_out
        self.mlp_hidden = mlp_hidden
        self.player_stat_dim = player_stat_dim
        self.dropout_p = dropout

        # Index 0 is the unknown-card token. Every card id shares this table.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Each card is its embedding plus one scalar, level / maxLevel.
        self.card_encoder = nn.Sequential(
            nn.Linear(embed_dim + 1, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_out),
            nn.ReLU(),
        )
        combined = encoder_out * 2 + player_stat_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def encode_deck(self, card_ids, levels):
        """Encode one deck. card_ids and levels both have shape (batch, 8)."""
        embedded = self.embedding(card_ids)
        cards = torch.cat([embedded, levels.unsqueeze(-1)], dim=-1)
        return self.card_encoder(cards).mean(dim=1)

    def forward(self, card_ids, levels, player_stats=None):
        """Return one logit per match, the log-odds that Player 1 wins.

        card_ids and levels have shape (batch, 2, 8): player axis, then cards.
        player_stats is required only for the with-player-stats configuration.
        """
        if card_ids.ndim != 3 or card_ids.shape[1:] != (2, 8):
            raise ValueError("card_ids must have shape (batch, 2, 8).")
        if levels.shape != card_ids.shape:
            raise ValueError("levels must have the same shape as card_ids.")
        if self.player_stat_dim == 0:
            if player_stats is not None:
                raise ValueError("deck-only model does not accept player statistics.")
        elif player_stats is None or player_stats.shape[-1] != self.player_stat_dim:
            raise ValueError(
                f"with-player-stats model expects player_stats with last dimension {self.player_stat_dim}."
            )
        decks = [self.encode_deck(card_ids[:, side], levels[:, side]) for side in (0, 1)]
        features = torch.cat(decks, dim=-1) if player_stats is None else torch.cat([*decks, player_stats], dim=-1)
        return self.mlp(features).squeeze(-1)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
