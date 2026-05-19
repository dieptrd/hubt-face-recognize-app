import os
from logger import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams, Distance, MultiVectorConfig, PointStruct
import uuid

from appSettings import settings
import commons

class Faces:
    def __init__(self, db):
        self.db = db
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
    
    def count(self):
        try:
            count = self.db.count(
                collection_name=self.collection_name
            )
            return count
        except Exception as e:
            logger.error("Error getting points count: %s", e)
            return 0
    
    def load_all_faces(self, payload_key, filter_list=None):
        offset = 0
        total = 0
        faces = []
        
        scroll_filter = None
        if filter_list is not None:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="payload_key",
                        match=models.MatchAny(any=filter_list),
                    ),
                ]
            )
        
        while offset != None:
            try:
                points, offset = self.db.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=True,
                )
                faces.extend(points)
                total += len(points)
            except Exception as e:
                logger.error("Error loading faces: %s", e)
                break
            finally:
                logger.info("Loaded faces, new offset: %s, total faces loaded: %s", offset, len(faces))

        return (faces, total)

    def close(self):
        self.db.close()
        return self
        
    def clear(self):
        try:
            self.db.delete_collection(collection_name=self.collection_name)
            logger.info("Collection '{}' cleared".format(self.collection_name))
        except Exception as e:
            logger.error("Error clearing DB: %s", e)
        return self

    def upsert_faces(self, points):
        try: 
            self.db.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            logger.info("Upserted faces to DB")
            return True
        except Exception as e:
            logger.error("Error upserting face to DB: %s", e)
            return False

    def get_face(self, point_id):
        try:
            points = self.db.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            if not points:
                raise ValueError(f"Point with ID {point_id} does not exist.")
            return points[0]
        except Exception as e:
            logger.error("Error inserting vector to DB: %s", e)
        return None

    def add_vector(self, id, vector):
        try: 
            point = self.get_face()
            vectors = commons._safe_get(point, "vectors") if point is not None else []
            payload = commons._safe_get(point, "payload") if point is not None else {}
            vectors.extend(vector)
            self.db.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    PointStruct(
                        id= id,
                        vector= vectors,
                        payload= payload
                    )
                ]
            )
            logger.info("Inserted vector for face {} to DB".format(id))
            return True
        except Exception as e:
            logger.error("Error inserting vector to DB: %s", e)
        return False


class DbProvider:
    def __init__(self):
        self.host = None
        self.client = None
        self.db = None
    
    def reload_db(self, clear_client=False):
        self.host = settings.get("VECTORDB","HOST", fallback= "localhost")
        self.port = settings.getint("VECTORDB","PORT", fallback= 6333)
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096) 
        # Close existing DB connection if any
        self.close_db() 
        if clear_client:
            self.clear_client()
        return self
    
    def get_client(self, key="collection"):
        if self.host is None:
            self.reload_db()
            
        if self.client is None:
            client_path = os.path.join("./vectordb","client") 
            self.client = QdrantClient(path=client_path)
            try:
                self.client.get_collection(collection_name=self.collection_name)
                print("Client collection '{}' exists".format(self.collection_name))
            except Exception:
                print("Client collection '{}' not found, creating...".format(self.collection_name))
                self.client.create_collection(
                    collection_name= self.collection_name,
                    vectors_config= VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE,
                        # multivector_config=models.MultiVectorConfig(
                        #     comparator=models.MultiVectorComparator.MAX_SIM
                        # )
                    ),
                )
            return self.client
        return self.client
    
    def clear_client(self):
        client = self.get_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
            print("Local collection '{}' cleared".format(self.collection_name))
        except Exception as e:
            print("Error clearing local DB: ", e)
        self.client = None
        
    def get_all_faces_client(self):
        client = self.get_client()
        if client is None:
            return []
        points, _ = client.scroll(
            collection_name=self.collection_name,
            offset=0,
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )
        return points

    def get_db(self):
        try: 
            if self.host is None:
                self.reload_db()
                
            if self.db is None: 
                self.db = QdrantClient(self.host, port=self.port)
            # Ensure remote collection exists; create it if missing
            try:
                self.db.get_collection(collection_name=self.collection_name)
                print("Remote collection '{}' exists".format(self.collection_name))
            except Exception:
                print("Remote collection '{}' not found, creating...".format(self.collection_name))
                self.db.create_collection(
                    collection_name= self.collection_name,
                    vectors_config= VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE,
                        # multivector_config=models.MultiVectorConfig(
                        #     comparator=models.MultiVectorComparator.MAX_SIM
                        # )
                    ),
                )
            return self.db
        except Exception as e:
            print("Error connecting to DB: ", e)
            return None

    def get_points_count_client(self):
        client = self.get_client()
        if client is None:
            return 0
        try:
            count = client.count(
                collection_name=self.collection_name
            )
            return count
        except Exception as e:
            logger.error("Error getting points count: %s", e)
            return 0

    def close_db(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def load_all_faces_to_client_with_filter(self, key="msv", filter_list=None):
        """Load faces from remote Qdrant DB and upsert to local in-memory Qdrant DB for fast access
            kwargs:
                filter_list: list of values to filter by, default is ["undefined", "TH14.01"]
        """
        db = self.get_db()
        client = self.get_client()
        offset = 0
        total = 0
        
        scroll_filter = None
        if filter_list is not None:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=filter_list),
                    ),
                ]
            )
        if db is None or client is None:
            return 0
        
        while offset != None:
            try:
                points, offset = db.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=True,
                )
                
                client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=points
                )
                total += len(points)
            except Exception as e:
                logger.error("Error loading faces to client: %s", e)
                break
            finally:
                logger.info("Upserted points, new offset: %s, total points upserted: %s", offset, len(points))

        return total
    
    def update_face_client(self, face_id, payload):
        client = self.get_client()
        if client is None:
            return False
        try:
            existing_point = client.get(
                collection_name=self.collection_name,
                id=face_id,
                with_payload=True,
                with_vector=False,
            )
            if existing_point is not None:
                updated_point = PointStruct(
                    id=face_id,
                    payload=payload,
                    vector=existing_point.vector
                )
                client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=[updated_point]
                )
                print("Updated face {} in local DB".format(face_id))
                return True
            else:
                print("Face ID {} not found in local DB".format(face_id))
                return False
        except Exception as e:
            print("Error updating face in local DB: ", e)
            return False
        
    def upsert_face_db(self, points):
        db = self.get_db()
        if db is None:
            return False
        try: 
            db.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            print("Upserted faces to remote DB")
            return True
        except Exception as e:
            print("Error upserting face to remote DB: ", e)
            return False
        
    def upsert_face_client(self, id, vector, payload):
        client = self.get_client()
        if client is None:
            return False
        try: 
            client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    PointStruct(
                        id=id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            print("Upserted face to local DB")
            return True
        except Exception as e:
            print("Error upserting face to local DB: ", e)
            return False
        
db = DbProvider()