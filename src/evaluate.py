import numpy as np

from src.tools import get_rhymes, count_syllables, get_shared_retriever


def rhyme_density(text: str) -> float:
    """
    Estimate rhyme density by checking if the last words of lines rhyme.
    Returns a score between 0 and 1.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return 0.0

    rhyme_pairs = 0
    for i in range(len(lines) - 1):
        words_i = lines[i].split()
        if not words_i:
            continue
        last_word_i = words_i[-1].strip('.,!?()').lower()

        for j in range(i + 1, len(lines)):
            words_j = lines[j].split()
            if not words_j:
                continue
            last_word_j = words_j[-1].strip('.,!?()').lower()

            if last_word_j in get_rhymes(last_word_i):
                rhyme_pairs += 1
                break

    return rhyme_pairs / (len(lines) - 1)


def syllable_consistency(text: str) -> float:
    """
    Calculate syllable consistency.
    Returns 1.0 for perfectly consistent lines, lower for high variance.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return 1.0

    counts = [count_syllables(line) for line in lines]
    mean_syll = np.mean(counts)
    if mean_syll == 0:
        return 0.0

    std_dev = np.std(counts)
    cv = std_dev / mean_syll
    return 1.0 / (1.0 + cv)


def originality_score(generated_text: str) -> float:
    """
    Compare generated lines against the corpus.
    Returns (1.0 - max_similarity).
    """
    retriever = get_shared_retriever()

    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    if not lines:
        return 1.0

    if retriever.index is None:
        retriever.load_index()

    gen_embeddings = retriever.model.encode(lines).astype('float32')

    max_sim = 0.0
    for emb in gen_embeddings:
        norm_emb = emb / np.linalg.norm(emb)
        dist, _ = retriever.index.search(norm_emb.reshape(1, -1), 1)
        # IndexFlatIP with normalized vectors returns cosine similarity directly.
        sim = dist[0][0]
        max_sim = max(max_sim, sim)

    return 1.0 - max_sim
