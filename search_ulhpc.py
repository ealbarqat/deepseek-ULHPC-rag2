import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

torch.set_num_threads(1)  # Safe for macOS

# Timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"ulhpc_search_log_{timestamp}.txt"

# Load embedding model
print("Loading model: all-mpnet-base-v2 ...")
model = SentenceTransformer("all-mpnet-base-v2")

# Load FAISS index and metadata
print("Loading FAISS index and metadata...")
index = faiss.read_index("ulhpc_faiss.index")
print("Index has", index.ntotal, "vectors.")

with open("ulhpc_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Query loop
print("\nAsk your question about ULHPC (type 'exit' to quit):\n")
while True:
    query = input(" You: ").strip()
    if query.lower() == "exit":
        break

    if not query:
        print(" Please enter a valid question.")
        continue

    # Step 1: Embed query
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    print("Query vector shape:", query_vector.shape)

    # Step 2: Search in FAISS
    top_k = 5
    score_threshold = 0.55  # Ignore weak matches
    scores, indices = index.search(query_vector, k=top_k)
    print("Debug → indices:", indices)
    print("Scores:", scores)

    # Step 3: Print and log results
    print("\n Top Matches:")
    context_blocks = []
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n Question: {query}\n")
        log.write(" Top Matches:\n")

        for rank, idx in enumerate(indices[0]):
            score = scores[0][rank]
            if score < score_threshold:
                continue

            title = metadata[idx]["title"]
            content = metadata[idx]["content"]

            # Display
            print(f"\n#{rank + 1}  {title}")
            print(f"Score: {score:.4f}")
            print(" Content Preview:")
            print(content[:500].strip() + "...")
            print("-" * 60)

            # Log
            log.write(f"\n#{rank + 1}  {title}\n")
            log.write(f"Score: {score:.4f}\n")
            log.write(content[:500].strip() + "...\n")
            log.write("-" * 60 + "\n")

            # For RAG prompt
            context_blocks.append(content.strip())

        # Step 4: Build a RAG prompt
        rag_prompt = "\n\n".join(context_blocks)
        final_prompt = f"### Context:\n{rag_prompt}\n\n### Question:\n{query}\n\n### Answer:"
        log.write("\n Final RAG Prompt:\n" + final_prompt + "\n")
        print("\n Final RAG Prompt:\n" + final_prompt)
        # Auto-copy prompt to clipboard (for macOS)
try:
    import pyperclip
    pyperclip.copy(final_prompt)
    print(" Final prompt copied to clipboard — ready to paste into Page Assist!")
except ImportError:
    print(" pyperclip not installed. Run: pip install pyperclip")

