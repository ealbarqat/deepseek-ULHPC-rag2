import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
torch.set_num_threads(1)  # Limit CPU threads (safe on Mac)

# Step 1: Load chunks
with open("ulhpc_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["content"] for chunk in chunks]
titles = [chunk["title"] for chunk in chunks]

# Step 2: Load nomic embedding model
print("Loading model: all-mpnet-base-v2 ...")
#model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
model = SentenceTransformer("all-mpnet-base-v2")

# Step 3: Generate embeddings
print("Embedding", len(texts), "chunks...")
embeddings = model.encode(
    texts,
    batch_size=4,                  #  Reduce batch size (try 4–8 max)
    show_progress_bar=True,
    normalize_embeddings=True
)
embeddings_np = np.array(embeddings).astype("float32")

# Step 4: Create FAISS index (L2 for normalized vectors is same as cosine)
print("Building FAISS index...")
index = faiss.IndexFlatIP(embeddings_np.shape[1])  # IP = inner product (cosine similarity)
index.add(embeddings_np)

# Step 5: Save index and metadata
faiss.write_index(index, "ulhpc_faiss.index")
with open("ulhpc_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Indexed {len(chunks)} chunks using all-mpnet-base-v2 embeddings. Saved to disk.")
