"""Model definition for the LyricLSTM language model.

This module contains the PyTorch implementation of a word-level LSTM
for song lyrics generation.
"""

import torch
import torch.nn as nn


class LyricLSTM(nn.Module):
    """A word-level LSTM model for song lyrics generation.

    The model uses an embedding layer followed by LSTM layers and a linear
    output layer to predict the next word in a sequence.
    """

    def __init__(self, word_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout_prob: float = 0.2) -> None:
        """Initialize the LyricLSTM model.

        Args:
            word_size: Size of the vocabulary.
            embed_dim: Dimension of word embeddings.
            hidden_dim: Dimension of LSTM hidden state.
            num_layers: Number of LSTM layers.
            dropout_prob: Dropout probability for regularization.
        """
        super().__init__()
        self.embedding = nn.Embedding(word_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.ffn = nn.Linear(hidden_dim, word_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length).

        Returns:
            Output tensor of shape (batch_size, word_size) representing
            logits for each word in the vocabulary.
        """
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout(out)
        return self.ffn(out)