import requests
from bs4 import BeautifulSoup
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# ----------------------------
# Configuration
# ----------------------------
BASE_URL = "https://docs.frappe.io/erpnext/"
SAVE_PATH = "vector_store"
INDEX_FILE = os.path.join(SAVE_PATH, "erpnext.index")
META_FILE = os.path.join(SAVE_PATH, "metadata.pkl")

os.makedirs(SAVE_PATH, exist_ok=True)

# ----------------------------
# Fetch and Parse Page
# ----------------------------
def fetch_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

print("Fetching documentation...")
text = fetch_page(BASE_URL)

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)

print(f"Total chunks created: {len(chunks)}")

# ----------------------------
# Embeddings
# ----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(chunks)

# ----------------------------
# FAISS Index
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

# Save metadata
with open(META_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("✅ Index created successfully.")
