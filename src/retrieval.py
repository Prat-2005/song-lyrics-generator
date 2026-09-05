import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

from src.config import INDEX_PATH, METADATA_PATH, RETRIEVAL_MODEL_NAME, HF_TOKEN, DATA_PATH


class LyricsRetriever:
    def __init__(self, model_name=RETRIEVAL_MODEL_NAME):
        if HF_TOKEN:
            try:
                login(token=HF_TOKEN)
            except Exception as e:
                print(f"Warning: HuggingFace login failed: {e}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def _load_lyrics_dataframe(self):
        """Load every artist CSV in DATA_PATH directly — this is the dataset,
        there's no separate cache layer to read from anymore."""
        csv_files = sorted(Path(DATA_PATH).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in DATA_PATH: {DATA_PATH}. "
                "Add artist lyric CSVs there, or fetch one with scripts/fetch_lyrics.py."
            )

        frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Some CSV exports carry a leftover unnamed index column — drop it.
            df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
            frames.append(df[['Artist', 'Lyric']])

        return pd.concat(frames, ignore_index=True)

    def build_index(self):
        """Load lyrics, chunk them, embed, and save FAISS index."""
        print("Loading and chunking lyrics...")
        df = self._load_lyrics_dataframe()

        all_chunks = []
        all_artists = []

        for _, row in df.iterrows():
            artist = row.get('Artist', '')
            if pd.isna(artist):
                artist = ''
            artist = str(artist).strip()
            lyric = str(row['Lyric'])

            # Split by standard newlines or <NEWLINE> tags
            lines = lyric.replace(" <NEWLINE> ", "\n").split("\n")

            for line in lines:
                line = line.strip()
                if len(line) > 10:  # Ignore very short fragments
                    all_chunks.append(line)
                    all_artists.append(artist)

        self.metadata = [
            {"artist": artist, "text": text}
            for artist, text in zip(all_artists, all_chunks)
        ]

        print(f"Embedding {len(all_chunks)} lines...")
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        embedded_text = np.array(embeddings).astype('float32')

        dimension = embedded_text.shape[1]
        faiss.normalize_L2(embedded_text)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embedded_text)

        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Index saved to {INDEX_PATH}")

    def load_index(self):
        """Load FAISS index and metadata from disk, building it if missing."""
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            print("Index loaded from disk.")
        else:
            print("Index not found. Building now...")
            self.build_index()

    def retrieve_similar_lines(self, query: str, artist: str = None, k: int = 5):
        """Retrieve top-k similar lines, optionally filtered by artist."""
        if self.index is None:
            self.load_index()

        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        search_k = k * 20 if artist else k
        _, indices = self.index.search(query_embedding, search_k)

        results = []
        for idx in indices[0]:
            if len(results) == k:
                break
            if idx == -1:
                continue

            meta = self.metadata[idx]
            meta_artist = str(meta.get('artist', '')).strip().lower()
            if artist and meta_artist != artist.lower():
                continue

            results.append(meta['text'])

        # If not found enough for a specified artist, expand the search.
        if len(results) < k and artist:
            _, indices = self.index.search(query_embedding, min(len(self.metadata), search_k * 5))

            for idx in indices[0]:
                if len(results) == k:
                    break
                if idx == -1:
                    continue

                meta = self.metadata[idx]
                meta_artist = str(meta.get('artist', '')).strip().lower()
                if artist and meta_artist != artist.lower():
                    continue
                if meta['text'] not in results:
                    results.append(meta['text'])

        return results
