import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import subprocess
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)  # Safe for macOS

# Load embedding model
print(" Loading model: all-mpnet-base-v2 ...")
model = SentenceTransformer("all-mpnet-base-v2")

# Load FAISS index and metadata
print(" Loading FAISS index and metadata...")
index = faiss.read_index("ulhpc_faiss.index")
print(f" Index loaded with {index.ntotal} vectors.")

with open("ulhpc_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Query loop
print("\n Ask your question about ULHPC (type 'exit' to quit):\n")
while True:
    query = input(" You: ").strip()
    if query.lower() == "exit":
        break

    if not query:
        print(" Please enter a valid question.")
        continue

    # Embed query
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

    # Search index
    top_k = 5
    score_threshold = 0.55
    scores, indices = index.search(query_vector, k=top_k)

    # Filter results and prepare context
    context_blocks = []
    for rank, idx in enumerate(indices[0]):
        score = scores[0][rank]
        if score < score_threshold:
            continue
        context_blocks.append(metadata[idx]["content"].strip())

    if not context_blocks:
        print(" No relevant context found.")
        continue

    # Final prompt
    rag_prompt = "\n\n".join(context_blocks)
    final_prompt = f"### Context:\n{rag_prompt}\n\n### Question:\n{query}\n\n### Answer:"

    # Copy to clipboard (optional)
    try:
        import pyperclip
        pyperclip.copy(final_prompt)
        print(" Prompt copied to clipboard.")
    except ImportError:
        pass

    #  Run with local LLM via Page Assist (Ollama / CLI)
    print(" Generating answer from model...\n")
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:1.5b", final_prompt],
            capture_output=True, text=True
        )
        print(" Answer:\n" + result.stdout.strip())
    except Exception as e:
        print(" Error calling model:", e)
