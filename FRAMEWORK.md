# SentenceTransformers Reference Guide

A cheat sheet and framework reference for using `sentence-transformers` in Python. 

### ℹ️ Core Information
* **Purpose**: Generates semantic vector embeddings for text or images.
* **Architecture**: Built on PyTorch and Hugging Face Transformers.
* **Input Limit**: Default max length is **512 tokens** per text.
* **Applications**: Semantic search, clustering, and RAG pipelines.

### 🛠️ Built-in Code Functions

#### 1. Model Loading
Loads a pre-trained model into memory.
```python
from sentence_transformers import SentenceTransformer

# Loads model to specified device (cpu, cuda, mps)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
```

#### 2. Vector Generation
Converts text strings into numeric vector arrays.
```python
embeddings = model.encode(
    ["Text one", "Text two"], 
    batch_size=32, 
    normalize_embeddings=True # Normalizes vectors to length 1
)
```

#### 3. Cosine Similarity
Calculates the semantic similarity matrix between vectors.
```python
from sentence_transformers import util

scores = util.cos_sim(embeddings[0], embeddings[1])
```

#### 4. Dot Product
Calculates dot product similarity (faster for normalized vectors).
```python
scores = util.dot_score(embeddings[0], embeddings[1])
```

#### 5. Semantic Search
Queries an embedding collection for top K matches.
```python
# Returns list of dicts with 'corpus_id' and 'score'
hits = util.semantic_search(query_emb, corpus_embs, top_k=5)
```

#### 6. Local Saving
Saves the downloaded model to a local directory.
```python
model.save('local_model_directory')
```


# FAISS Cheat Sheet

### ℹ️ Core Information
* **Purpose**: Fast dense vector similarity search, clustering, and nearest-neighbor retrieval.
* **Architecture**: Written in highly optimized C++ with seamless Python wrappers.
* **Scale**: Designed to search through millions or billions of vectors instantly.
* **Core Concept**: Trades a tiny fraction of accuracy for massive search speed via Approximate Nearest Neighbors (ANN).

### 🛠️ Built-in Code Functions

#### 1. Flat Index (Exact Search)
Creates a baseline index that performs exact brute-force Euclidean (L2) distance search.
```python
import faiss

# Dimension must match your embedding model (e.g., 384 for all-MiniLM-L6-v2)
dimension = 384
index = faiss.IndexFlatL2(dimension)
```

#### 2. Adding Vectors
Populates the index. FAISS strictly requires 2D NumPy arrays of type `float32`.
```python
import numpy as np

# Convert embeddings to float32 NumPy array
vectors = np.array(embeddings).astype('float32')

# Add to index
index.add(vectors)
print(index.ntotal) # Returns total vectors stored
```

#### 3. Searching the Index
Queries the index to find the `k` closest matching vectors.
```python
# Query vector must also be a 2D float32 array
query_vector = np.array([single_embedding]).astype('float32')

k = 5 # Number of top results to return
distances, indices = index.search(query_vector, k)

# 'indices' contains matching row IDs; 'distances' contains similarity scores
```

#### 4. Cosine Similarity Index
Uses inner product (Dot Product) search instead of L2 distance. 
```python
# For normalized vectors, Inner Product equals Cosine Similarity
index = faiss.IndexFlatIP(dimension)
```

#### 5. IVF Index (Fast Large-Scale Search)
Speeds up massive datasets by clustering the vector space and searching only the closest clusters.
```python
nlist = 100  # Number of clusters to build
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# IVF indexes require an explicit training step on your data distribution
index_ivf.train(vectors)
index_ivf.add(vectors)

# Set nprobe: higher values mean better accuracy but slower speed
index_ivf.nprobe = 10  # Search only 10 clusters out of 100
distances, indices = index_ivf.search(query_vector, k)
```

#### 6. Saving and Loading Indexes
Persists the index structure and stored vectors directly to the disk.
```python
# Save index to file
faiss.write_index(index, "my_vector_index.faiss")

# Load index from file
index = faiss.read_index("my_vector_index.faiss")
```
