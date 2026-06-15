"""Streamlit app for generating song lyrics using a trained LSTM model."""

import streamlit as st
import torch
from pathlib import Path
from src.model import LyricLSTM
from src.inference import load_model, generate_text
from src.utils import get_device, format_lyrics


@st.cache_resource
def load_model_cached():
    """Load the model with caching to avoid reloading on every run.

    Returns:
        Tuple containing the model, word mappings, and config.
    """
    device = get_device()
    checkpoint_path = Path("models/lyrics_lstm_checkpoint.pth")

    try:
        return load_model(checkpoint_path, device)
    except FileNotFoundError:
        st.error("Checkpoint not found at models/lyrics_lstm_checkpoint.pth. Please ensure the file exists.")
        st.stop()
    except RuntimeError as e:
        st.error(f"Checkpoint appears corrupted. Please re-export the model.\n\nDetails: {str(e)}")
        st.stop()


def main():
    """Main function for the Streamlit app."""
    # Page config
    st.set_page_config(page_title="Song Lyrics Generator", page_icon="🎤", layout="centered")

    # Load model
    try:
        model, word2idx, idx2word, config = load_model_cached()
        if torch.cuda.is_available():
            st.success("Running on GPU! 🚀")
        else:
            st.info("Running on CPU.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Title
    st.title("🎤 Song Lyrics Generator")

    # Sidebar controls
    st.sidebar.header("Generation Parameters")
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.05)
    top_k = st.sidebar.slider("Top-K", 5, 100, 20, 5)
    max_words = st.sidebar.slider("Max Words", 30, 250, 30, 10)

    # Main area
    prompt = st.text_input("Enter a prompt", placeholder="e.g. i am / love is / you better")

    if st.button("Generate Lyrics"):
        if not prompt.strip():
            st.warning("Please enter a prompt to generate lyrics.")
            return

        # Generate lyrics
        with st.spinner("Generating lyrics..."):
            try:
                device = get_device()
                raw_lyrics = generate_text(
                    model, word2idx, idx2word, prompt, max_words,
                    temperature, top_k, device
                )
                formatted_lyrics = format_lyrics(raw_lyrics)
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                return

        # Display output
        st.subheader("Generated Lyrics")
        st.code(formatted_lyrics, language=None)

        # Word count
        word_count = len(raw_lyrics.split())
        st.caption(f"Word count: {word_count}")


if __name__ == "__main__":
    main()