# deepseek-ULHPC-rag2
This repository contains a pipeline work for RAG system that is relying on all-mpnet-base-v2 embeding model and deepseek1.5b model and based on ULHPC docs.

# What you need to have? 
first have folder that contains files downloaded from here https://github.com/ULHPC/ulhpc-docs/tree/master 

# How to run it? 
- run first `python extract_md_chunks.py` this will generate file `ulhpc_chunks.json`
- run `python embed_and_store.py` this will generate file `ulhpc_faiss.index` and metadata file [Embed and index]
- run `python search_ulhpc.py` this will generate propmt and copy it to clipboard so it can be used with LLM, and log file for tracking
- run `python search_answer_only.py`this will generate answers for your question, it needs to run ollama model 1.5b [Full RAG with model]

# Requirements?
- RAM > 8 GB
- Python >= 3.9
- Ollama model > ollama pull deepseek-r1:1.5b
- sentence-transformers==2.2.2
- faiss-cpu==1.7.4
- numpy==1.24.4
- torch==2.0.1
- pyperclip==1.8.2

- file requirements.txt has the full list

# Setup: 
```bash
git clone https://github.com/yourname/ulhpc-rag.git
cd ulhpc-rag
pip install -r requirements.txt
