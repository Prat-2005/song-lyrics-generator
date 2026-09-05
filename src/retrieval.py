import os
import pickle
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from src.config import INDEX_PATH, METADATA_PATH, RETRIEVAL_MODEL_NAME, HF_TOKEN, CACHE_DIR

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
        if not CACHE_DIR.exists() or not any(CACHE_DIR.glob("*.json")):
            raise FileNotFoundError(f"No cached lyrics found in directory: {CACHE_DIR}. Please run src/fetch_lyrics.py first.")

        data = []
        for json_file in sorted(CACHE_DIR.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                data.append({
                    'Artist': entry.get('artist', ''),
                    'Lyric': entry.get('lyrics', '')
                })

        return pd.DataFrame(data)

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

            # For very long lines, we could further split by commas or periods,
            # but for lyrics, line-by-line is usually a good unit.
            for line in lines:
                line = line.strip()
                if len(line) > 10: # Ignore very short fragments
                    all_chunks.append(line)
                    all_artists.append(artist)

        self.metadata = [
            {"artist": artist, "text": text} 
            for artist, text in zip(all_artists, all_chunks)
        ]

        print(f"Embedding {len(all_chunks)} lines...")
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        embedded_text = np.array(embeddings).astype('float32')

        # Initialize FAISS index
        dimension = embedded_text.shape[1]
        faiss.normalize_L2(embedded_text)
        # Store the index as an inner product index for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embedded_text)

        # Persist to disk
        os.makedirs("models", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Index saved to {INDEX_PATH}")

    def load_index(self):
        """Load FAISS index and metadata from disk."""
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

        # Search for a reasonable initial batch
        search_k = k * 20 if artist else k
        _ , indices = self.index.search(query_embedding, search_k)

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

        # If not found enough songs and artist was specified, expand search
        if len(results) < k and artist:
            # Search a larger batch to find more of the specific artist
            _ , indices = self.index.search(query_embedding, min(len(self.metadata), search_k * 5))

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
