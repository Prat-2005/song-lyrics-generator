import streamlit as st

from src.agent import generate_lyrics


def main():
    """Main function for the Streamlit app."""
    # Page config
    st.set_page_config(page_title="Song Lyrics Generator", page_icon="🎤", layout="centered")

    # Custom CSS for a more polished look
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .lyrics-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #6c5ce7;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        font-family: 'Georgia', serif;
        font-size: 1.2rem;
        line-height: 1.6;
        color: #2d3436;
        white-space: pre-wrap;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("🎤 Song Lyrics Generator")
    st.markdown(
        "<p style='text-align: center; color: #636e72;'>"
        "Craft professional, artist-inspired lyrics with AI-driven precision</p>",
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    model_label = "LyricAI"

    with st.sidebar.expander("🎨 Generation Parameters", expanded=True):
        max_words = st.slider("Max Words", 80, 500, 120, 10)
        temperature = st.slider("Temperature (Creativity)", 0.5, 2.0, 1.0, 0.1)

    # Main area
    st.markdown("### ✍️ Composition")
    theme = st.text_input("Theme", placeholder="e.g. loneliness in a big city")

    col1, col2 = st.columns(2)
    with col1:
        artist_style = st.text_input("Artist Style (Optional)", placeholder="e.g. Drake")
    with col2:
        mood = st.selectbox(
            "Mood", options=["emotional", "upbeat", "dark", "romantic", "melancholic"], index=0
        )

    if not st.button("Generate Lyrics", use_container_width=True):
        return

    if not theme.strip():
        st.warning("Please enter a theme to generate lyrics.")
        return

    st.markdown("---")
    st.subheader("✨ Generated Lyrics")
    lyrics_placeholder = st.empty()

    def stream_callback(full_text):
        lyrics_placeholder.markdown(f'<div class="lyrics-card">{full_text}</div>', unsafe_allow_html=True)

    with st.spinner(f"Generating with {model_label}..."):
        try:
            raw_lyrics, generation_log, metrics = generate_lyrics(
                theme, artist_style, mood, max_words, temperature,
                stream_callback=stream_callback,
            )
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            return

    if raw_lyrics.startswith("Error:"):
        st.error(raw_lyrics)
        return

    lyrics_placeholder.markdown(f'<div class="lyrics-card">{raw_lyrics}</div>', unsafe_allow_html=True)

    # Evaluation scores
    st.markdown("### 📊 Quality Metrics")
    rhyme_score = metrics.get("rhyme_density", 0.0)
    syllable_score = metrics.get("syllable_consistency", 0.0)
    orig_score = metrics.get("originality_score", 0.0)

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Rhyme Density", f"{rhyme_score:.2%}")
    with m_col2:
        st.metric("Syllable Consistency", f"{syllable_score:.2%}")
    with m_col3:
        st.metric("Originality Score", f"{orig_score:.2%}")

    if generation_log:
        with st.expander("🛠️ View Generation Steps"):
            for entry in generation_log:
                st.write(f"**{entry['step']}:** {entry['info']}")

    # Word count and model info
    st.markdown("---")
    word_count = len(raw_lyrics.split())
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Word Count", word_count)
    with info_col2:
        st.metric("Model", model_label)
    with info_col3:
        st.metric("Mood", mood)


if __name__ == "__main__":
    main()
