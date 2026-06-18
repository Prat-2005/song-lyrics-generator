"""Streamlit app for generating song lyrics using trained models (LSTM or Transformer)."""

import streamlit as st
import torch
from pathlib import Path
from src.model import LyricLSTM, LyricTransformer
from src.inference import load_model, generate_text
from src.utils import get_device, format_lyrics


@st.cache_resource
def load_model_cached(model_type: str):
    """Load the model with caching to avoid reloading on every run.

    Args:
        model_type: Either "LSTM" or "Transformer"

    Returns:
        Tuple containing the model, word mappings, and config.
    """
    device = get_device()
    
    if model_type == "LSTM":
        checkpoint_path = Path("models/lyrics_lstm_checkpoint.pth")
    else:  # Transformer
        checkpoint_path = Path("models/best_transformer_model.pth")

    try:
        return load_model(checkpoint_path, device)
    except FileNotFoundError:
        st.error(f"Checkpoint not found at {checkpoint_path}. Please ensure the file exists.")
        st.stop()
    except RuntimeError as e:
        st.error(f"Checkpoint appears corrupted. Please re-export the model.\n\nDetails: {str(e)}")
        st.stop()


def main():
    """Main function for the Streamlit app."""
    # Page config
    st.set_page_config(page_title="Song Lyrics Generator", page_icon="🎤", layout="centered")

    # Title
    st.title("🎤 Song Lyrics Generator")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    # Model selection (like Claude's model dropdown)
    model_type = st.sidebar.radio(
        "Choose Model:",
        options=["LSTM", "Transformer"],
        help="LSTM (Fast) vs Transformer (Advanced)",
        index=0
    )
    
    # Load model based on selection
    try:
        model, word2idx, idx2word, config = load_model_cached(model_type)
        st.sidebar.success(f"✓ {model_type} model loaded")
        
        if torch.cuda.is_available():
            st.sidebar.info("🚀 Running on GPU")
        else:
            st.sidebar.info("💻 Running on CPU")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Generation parameters
    st.sidebar.header("Generation Parameters")
    max_words = st.sidebar.slider("Max Words", 80, 500, 120, 10)
    temperature = st.sidebar.slider("Temperature (Creativity)", 0.5, 2.0, 1.4, 0.1)
    top_k = st.sidebar.slider("Top-K", 5, 50, 20, 1)

    # Main area
    prompt = st.text_input("Enter a prompt", placeholder="e.g. i am / love is / you better")

    if st.button("Generate Lyrics", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt to generate lyrics.")
            return

        # Generate lyrics
        with st.spinner(f"Generating with {model_type}..."):
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

        # Word count and model info
        word_count = len(raw_lyrics.split())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", word_count)
        with col2:
            st.metric("Model", model_type)
        with col3:
            st.metric("Temperature", temperature)


if __name__ == "__main__":
    main()