import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("mock_posts")

data = collection.get(
    limit=5,
    include=["documents", "metadatas", "embeddings"]
)

# ids are always present in the response
print("IDS:", data["ids"])
print("\nDOCUMENT (first):\n", data["documents"][0])
print("\nMETADATA (first):\n", data["metadatas"][0])
print("\nEMBEDDING DIM:", len(data["embeddings"][0]))
print("\nFIRST 5 EMBEDDING VALUES:", data["embeddings"][0][:5])