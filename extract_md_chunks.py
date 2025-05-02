import os
import re
import json

def read_markdown_files_recursively(root_dir):
    """Read all .md files from a root directory recursively"""
    md_texts = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.md'):
                file_path = os.path.join(dirpath, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_texts.append((file_path, f.read()))
    return md_texts

def split_into_chunks(md_text, min_chunk_length=100):
    """Split markdown content into sections based on headings"""
    chunks = []
    sections = re.split(r'(?m)^#{1,6}\s+', md_text)

    for section in sections:
        lines = section.strip().split('\n')
        if not lines or len(lines[0].strip()) < 5:
            continue

        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        if len(content) < min_chunk_length:
            continue

        chunks.append({
            "title": title,
            "content": content
        })

    return chunks

def extract_chunks_from_directory(root_dir):
    all_chunks = []
    files = read_markdown_files_recursively(root_dir)

    for file_path, text in files:
        file_chunks = split_into_chunks(text)
        all_chunks.extend(file_chunks)

    print(f"Extracted {len(all_chunks)} chunks from {len(files)} markdown files.")
    return all_chunks

if __name__ == "__main__":
    # Since script and md files are in the same directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    chunks = extract_chunks_from_directory(root_dir)

    # Save to JSON for inspection
    with open("ulhpc_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Saved to ulhpc_chunks.json")
