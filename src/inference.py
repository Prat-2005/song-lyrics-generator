"""Inference functions for the lyrics generator."""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Any, Union
from src.model import LyricLSTM, LyricTransformer


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[Union[LyricLSTM, LyricTransformer], Dict[str, int], Dict[int, str], Dict[str, Any]]:
    """Load the trained model from a checkpoint file.

    Supports both LSTM and Transformer checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple containing:
        - model: The loaded model (LyricLSTM or LyricTransformer) in evaluation mode.
        - word2idx: Dictionary mapping words to indices.
        - idx2word: Dictionary mapping indices to words.
        - config_dict: Dictionary with model hyperparameters.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint is missing required keys.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    required_keys = ["model_state_dict", "word2idx", "idx2word"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")
    
    # Validate checkpoint structure
    if not isinstance(checkpoint.get("word2idx"), dict):
        raise RuntimeError("word2idx in checkpoint is not a dictionary")
    if not isinstance(checkpoint.get("idx2word"), dict):
        raise RuntimeError("idx2word in checkpoint is not a dictionary")

    # Detect model type (Transformer vs LSTM)
    is_transformer = "d_model" in checkpoint or "vocab_size" in checkpoint
    
    if is_transformer:
        # Load Transformer model
        vocab_size = checkpoint.get("vocab_size", len(checkpoint["word2idx"]))
        d_model = checkpoint.get("d_model", 128)
        nhead = checkpoint.get("nhead", 4)
        ff_dim = checkpoint.get("ff_dim", 1024)
        num_layers = checkpoint.get("num_layers", 2)
        dropout = checkpoint.get("dropout", 0.2)
        seq_len = checkpoint.get("seq_len", 512)

        model = LyricTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        config_dict = {
            "model_type": "Transformer",
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "seq_len": seq_len
        }
    else:
        # Load LSTM model (backward compatibility)
        word_size = checkpoint.get("word_size", len(checkpoint["word2idx"]))
        embed_dim = checkpoint.get("embed_dim", 128)
        hidden_dim = checkpoint.get("hidden_dim", 512)
        num_layers = checkpoint.get("num_layers", 2)
        dropout_prob = checkpoint.get("dropout_prob", 0.2)

        model = LyricLSTM(
            word_size=word_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )

        config_dict = {
            "model_type": "LSTM",
            "word_size": word_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_prob": dropout_prob
        }

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["word2idx"], checkpoint["idx2word"], config_dict


def generate_text(model: Union[LyricLSTM, LyricTransformer], word2idx: Dict[str, int], idx2word: Dict[int, str],
                  prompt: str, max_words: int, temperature: float, top_k: int, device: torch.device,
                  seq_length: int = 20) -> str:
    """Generate text using the trained model.

    Supports both LSTM and Transformer models.

    Args:
        model: The trained model (LyricLSTM or LyricTransformer).
        word2idx: Dictionary mapping words to indices.
        idx2word: Dictionary mapping indices to words.
        prompt: Initial text prompt for generation.
        max_words: Maximum number of words to generate.
        temperature: Temperature for sampling (higher = more random).
        top_k: Number of top tokens to consider for sampling.
        device: Device to run the model on.
        seq_length: Sequence length to use (default 20 for LSTM, adjustable for Transformer).

    Returns:
        str: The prompt followed by generated text.
    """
    # Process prompt
    tokens = prompt.lower().split()

    # Convert tokens to indices
    indices = []
    for token in tokens:
        if token in word2idx:
            indices.append(word2idx[token])
        else:
            indices.append(word2idx.get("<UNK>", 0))  # Default to 0 if <UNK> not found

    # Pad or truncate to sequence length
    if len(indices) < seq_length:
        # Left-pad with <PAD> (index 0)
        indices = [0] * (seq_length - len(indices)) + indices
    else:
        # Truncate to last seq_length tokens
        indices = indices[-seq_length:]

    # Generate text
    generated_indices = indices.copy()

    # Get special token indices
    pad_idx = word2idx.get("<PAD>", 0)
    unk_idx = word2idx.get("<UNK>", 0)

    # Determine if model is Transformer (has 'transformer' attribute)
    is_transformer = hasattr(model, 'transformer')

    for _ in range(max_words):
        # Prepare input tensor
        input_tensor = torch.tensor([generated_indices[-seq_length:]], device=device)

        # Forward pass
        with torch.no_grad():
            if is_transformer:
                # Generate causal mask for Transformer
                seq_len = input_tensor.size(1)
                mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                logits = model(input_tensor, mask=mask)
            else:
                logits = model(input_tensor)

        # Apply temperature scaling
        logits = logits / temperature

        # Set logits for <PAD> and <UNK> to -inf
        logits[0, pad_idx] = float('-inf')
        logits[0, unk_idx] = float('-inf')

        # Apply top-K sampling
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        mask_tokens = torch.zeros_like(logits, dtype=torch.bool)
        mask_tokens.scatter_(1, top_k_indices, True)
        logits[~mask_tokens] = float('-inf')

        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token_idx = torch.multinomial(probs, 1).item()

        # Append to sequence
        generated_indices.append(next_token_idx)

    # Convert indices back to words
    generated_tokens = []

    for idx in generated_indices:
        if idx == pad_idx:
            continue

        generated_tokens.append(
            idx2word.get(idx, "<UNK>")
    )
        
    generated_words_only = generated_tokens[len(prompt.split()):]

    # Return prompt + generated text as a clean string
    return (
        prompt
        + " "
        + " ".join(generated_words_only)
    )