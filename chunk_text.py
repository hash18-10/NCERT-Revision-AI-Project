import os
import re

CHUNK_SIZE = 500
OVERLAP = 50
OUTPUT_FOLDER = "chunks"

def aggressive_recursive_split(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    sentences = re.split(r'(?<=[.!?]) +', text)
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)

        if current_length + sentence_length <= chunk_size:
            current_chunk.extend(sentence_words)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if overlap else []
            current_chunk = overlap_words + sentence_words
            current_length = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

if __name__ == "__main__":
    # Load text file
    with open("Understanding Media.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Create chunks
    chunks = aggressive_recursive_split(text)
    print(f"✅ Created {len(chunks)} chunks")

    # Create folder if not exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save chunks to separate files
    for i, c in enumerate(chunks, 1):
        file_path = os.path.join(OUTPUT_FOLDER, f"chunk_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(c)
        print(f"💾 Saved {file_path}")

    print("🎯 All chunks saved in the 'chunks' folder")
