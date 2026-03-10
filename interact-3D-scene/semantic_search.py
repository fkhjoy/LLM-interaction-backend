from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

objects = "front_door,rear_door,engine, wheel_front_left, wheel_front_right"

def search_object(query="", objects: str = ""):
    objects = list(objects.split(","))
    embeddings = model.encode(objects)

    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings))

    query = model.encode([query])

    D,I = index.search(query,1)

    return objects[I[0][0]]

if __name__ == "__main__":
    print(search_object("Show me the left wheel from the front", objects))