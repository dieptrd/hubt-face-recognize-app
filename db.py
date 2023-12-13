from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
class FaceRecognize:
    def load_faces(self):
        vector_size = 2622
        collection_name="hubt_faces"
        db = QdrantClient("localhost", port=6333)
        offset = 0

        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name= collection_name,
            vectors_config= VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        while offset != None:
            points, offset = db.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="class",
                            match=models.MatchAny(any=["undefined", "TH14.01"]),
                        ),
                    ]
                ),
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=True,
            )
            p = points[0]
            print(p.payload["img"])
            self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            print(offset)

d = FaceRecognize()
d.load_faces()