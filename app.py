from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

Collection_Name = 'MyCollection'

client = QdrantClient(':memory:')
model = SentenceTransformer("all-MiniLM-L6-v2")

if client.collection_exists(collection_name=Collection_Name):
    print("Collection Exists")
else:
    client.create_collection(
        collection_name= Collection_Name,
        vectors_config= VectorParams(
            distance = Distance.COSINE,
            size = model.get_embedding_dimension()
        )
    )


raw_text = [
    "Hey there, this is a sample data",
    "langchain is used for AI",
    "langchain has prompt templates"
]


embeddings = model.encode(raw_text)


points = [
    PointStruct(
        id = idx,
        vector=embedding.tolist(),
        payload={"text": text}
    )
    for idx, (text, embedding) in enumerate(zip(raw_text, embeddings))
]

client.upsert(collection_name = Collection_Name, points= points)


query = "what is the use of langchain?"

query_vector = model.encode(query)

search_results = client.query_points(
    collection_name= Collection_Name, query = query_vector, limit = 2
)

for i in search_results.points:
    print(f'Score : {i.score}')
    print(f'Text : {i.payload['text']}')
