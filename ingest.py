
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Read document
file_path = Path("employee_handbook_rag_sample.txt")
text = file_path.read_text(encoding="utf-8")

# Split document into chunks
chunk_size = 700
overlap = 100

chunks = [
    text[i:i + chunk_size].strip()
    for i in range(0, len(text), chunk_size - overlap)
]
chunks = [chunk for chunk in chunks if chunk]

print("Number of chunks:", len(chunks))

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save index and chunks
faiss.write_index(index, "document.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("RAG document ingestion completed successfully!")