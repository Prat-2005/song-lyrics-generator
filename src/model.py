"""Model definitions for lyrics generation.

This module contains PyTorch implementations of:
- LyricLSTM: Word-level LSTM for song lyrics generation
- LyricTransformer: Transformer-based model for song lyrics generation
"""

import torch
import torch.nn as nn
import math


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


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models.
    
    Uses sine and cosine functions of different frequencies to encode
    the position information in the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model.
            max_len: Maximum sequence length.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]


class LyricTransformer(nn.Module):
    """A Transformer-based model for song lyrics generation.

    The model uses an embedding layer followed by Transformer encoder layers
    and a linear output layer to predict the next word in a sequence.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, ff_dim: int = 1024, dropout: float = 0.2) -> None:
        """Initialize the LyricTransformer model.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: Dimension of the model (embedding and transformer).
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            ff_dim: Dimension of the feedforward network.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length).
            mask: Optional attention mask for autoregressive decoding.

        Returns:
            Output tensor of shape (batch_size, vocab_size) representing
            logits for each word in the vocabulary.
        """
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        x = x[:, -1, :]
        x = self.fc(x)
        return x