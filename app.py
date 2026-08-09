import streamlit as st
from src.agent import generate_lyrics_v2
from src.evaluate import rhyme_density, syllable_consistency, originality_score


def main():
    """Main function for the Streamlit app."""
    # Page config
    st.set_page_config(page_title="Song Lyrics Generator", page_icon="🎤", layout="centered")


    # Title
    st.title("🎤 Song Lyrics Generator")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    model_type = "AI Agent (Local LLM + Retrieval)"

    # Generation parameters
    st.sidebar.header("Generation Parameters")
    max_words = st.sidebar.slider("Max Words", 80, 500, 120, 10)
    temperature = st.sidebar.slider("Temperature (Creativity)", 0.5, 2.0, 1.0, 0.1)
    top_k = st.sidebar.slider("Top-K", 5, 50, 20, 1)

    # Main area
    theme = st.text_input("Theme", placeholder="e.g. loneliness in a big city")
    col1, col2 = st.columns(2)
    with col1:
        artist_style = st.text_input("Artist Style (Optional)", placeholder="e.g. Drake")
    with col2:
        mood = st.selectbox("Mood", options=["emotional", "upbeat", "dark", "romantic", "melancholic"], index=0)

    if st.button("Generate Lyrics", use_container_width=True):
        if not theme.strip():
            st.warning("Please enter a theme to generate lyrics.")
            return

        st.subheader("Generated Lyrics")
        lyrics_placeholder = st.empty()

        def stream_callback(_token, full_text):
            lyrics_placeholder.code(full_text, language="text")

        # Generate lyrics
        with st.spinner(f"Generating with {model_type}..."):
            try:
                raw_lyrics, tool_logs = generate_lyrics_v2(
                    theme, artist_style, mood, max_words, temperature, 
                    stream_callback=stream_callback
                )
                formatted_lyrics = raw_lyrics # Agent already returns formatted text
                # Ensure final output is written
                lyrics_placeholder.code(formatted_lyrics, language="text")
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                return

        # Evaluation scores
        st.markdown("📊 Quality Metrics")
        rhyme_score = rhyme_density(formatted_lyrics)
        syllable_score = syllable_consistency(formatted_lyrics)
        orig_score = originality_score(formatted_lyrics)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Rhyme Density", f"{rhyme_score:.2%}")
        with m_col2:
            st.metric("Syllable Consistency", f"{syllable_score:.2%}")
        with m_col3:
            st.metric("Originality Score", f"{orig_score:.2%}")

        if tool_logs:
            with st.expander("🛠️ View Agent Tool Logs"):
                for log in tool_logs:
                    st.write(f"**Tool:** `{log['tool']}`")
                    st.write(f"**Args:** `{log['args']}`")
                    st.write(f"**Result:** `{log['result']}`")
                    st.divider()

        # Word count and model info
        word_count = len(raw_lyrics.split())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", word_count)
        with col2:
            st.metric("Model", model_type)
        with col3:
            st.metric("Mood", mood)


if __name__ == "__main__":
    main()