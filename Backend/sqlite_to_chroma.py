import sqlite3
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DB_PATH = "store_social_data.db"
CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "mastodon_posts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
SELECT id, content, account_id, account_username, account_acct,
       tags, reblogs_count, favourites_count, replies_count,
       visibility, language
FROM posts
""")

rows = cursor.fetchall()
conn.close()

ids = []
documents = []
metadatas = []

for (pid, content, account_id, account_username, account_acct,
     tags, reblogs_count, favourites_count, replies_count,
     visibility, language) in rows:
    text = (content or "").strip()
    if not text:
        continue

    ids.append(str(pid))
    documents.append(text)
    metadatas.append({
        "account_id": account_id or "",
        "account_username": account_username or "",
        "account_acct": account_acct or "",
        "reblogs_count": reblogs_count or 0,
        "favourites_count": favourites_count or 0,
        "replies_count": replies_count or 0,
        "visibility": visibility or "public",
        "language": language or "",
        "tags": tags or "[]",   # JSON string
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