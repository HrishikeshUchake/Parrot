import sqlite3
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DB_PATH = "store_social_data.db"
CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "mock_posts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
SELECT id, title, body, tags, reactions, views, userId
FROM posts
""")

rows = cursor.fetchall()
conn.close()

ids = []
documents = []
metadatas = []

for (pid, title, body, tags, reactions, views, userId) in rows:
    title = title or ""
    body = body or ""
    text = (title + "\n\n" + body).strip()

    if not text:
        continue

    ids.append(str(pid))
    documents.append(text)

    metadatas.append({
        "userId": userId,
        "views": views,
        "has_reactions": reactions is not None,
        "tags": tags  # already JSON text
    })

print(f"Loaded {len(documents)} posts from SQLite.")

# 2) Generate embeddings (local)
model = SentenceTransformer(EMBEDDING_MODEL)

embeddings = model.encode(
    documents,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()

# 3) Create / load ChromaDB
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# 4) Upsert into ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Step B complete: embeddings stored in ChromaDB.")
print("Chroma count:", collection.count())