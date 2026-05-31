import os
from logger import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    Query,
    VectorParams,
    MultiVectorConfig,
    MultiVectorComparator,
    QuantizationConfig,
    ScalarQuantization,
    ScalarType,
    HnswConfigDiff,
    PointStruct
)
import uuid

from appSettings import settings
import commons

class Faces:
    def __init__(self, db:QdrantClient = None):
        self.db = db
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        
    @property
    def is_connected(self) -> bool:
        """Returns True if the client is initialized and reachable."""
        if self.db is None:
            return False
        try:
            # Quick lightweight check
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def count(self):
        try:
            result = self.db.count(
                collection_name=self.collection_name
            )
            return result.count
        except Exception as e:
            logger.error("Error getting points count: %s", e)
            return 0
    
    def load_all_faces(self, payload_key:str ="msv", filter_list=None):
        offset = 0
        total = 0
        faces = []
        
        scroll_filter = None
        if filter_list is not None:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=payload_key,
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
        if self.is_connected:
            self.db.close()
            self.db = None
        return self
        
    def clear(self):
        try:
            self.db.delete_collection(collection_name=self.collection_name)
            logger.info("Collection '{}' cleared".format(self.collection_name))
        except Exception as e:
            logger.error("Error clearing DB: %s", e)
        return self

    def upsert_face(self, id, vector, payload): 
        def is_2d_list(v):
            # Kiểm tra biến v có phải là list, không rỗng, và phần tử đầu tiên cũng là list
            return isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)
        try: 
            self.db.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=id,
                        vector= vector if is_2d_list(vector) else [vector],
                        payload = payload
                    )
                ],
            )
            print("Upserted 01 face to local DB")
            return True
        except Exception as e:
            print("Error upserting face to local DB: ", e)
            return False

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
            point = self.get_face(id)
            vectors = commons._safe_get(point, "vector", default=[]) if point is not None else []
            payload = commons._safe_get(point, "payload", default={}) if point is not None else {}
            print(f"persion {id} had: {len(vectors)} vectors.")
            vectors.append(vector)
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

    def find(self, vector, score_threshold=0.7):
        return self.db.query_points(
            collection_name= self.collection_name, 
            query=[vector],
            with_vectors=True,
            with_payload=True,
            limit= 1,
            score_threshold = score_threshold
        )

class RemoteFaces(Faces):
    def __init__(self):
        super().__init__(None)
        self.reload() 
        
    def reload(self): 
        self.close()
        self.host = settings.get("VECTORDB","HOST", fallback= "localhost")
        self.port = settings.getint("VECTORDB","PORT", fallback= 6333)
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096) 
        try:  
            db = self.db = QdrantClient(self.host, port=self.port)
            # Ensure remote collection exists; create it if missing
            try:
                db.get_collection(collection_name=self.collection_name)
                print("Remote collection '{}' exists".format(self.collection_name))
            except Exception:
                print("Remote collection '{}' not found, creating...".format(self.collection_name))
                db.create_collection(
                    collection_name= self.collection_name,
                    vectors_config= VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                    # # TỐI ƯU RAM & TỐC ĐỘ: Bật Scalar Quantization (Nén về int8)
                    # # Giúp giảm 4 lần dung lượng RAM lưu trữ vector nhưng giữ nguyên ~99% độ chính xác
                    # quantization_config=QuantizationConfig(
                    #     scalar=ScalarQuantization(
                    #         type=ScalarType.INT8,
                    #         quantile=0.99, # Giữ lại 99% phân phối dữ liệu để tránh mất mát độ chính xác
                    #         always_ram=True # Giữ mảng nén trên RAM để tìm kiếm cực nhanh
                    #     )
                    # ),
                    # TỐI ƯU ĐỘ CHÍNH XÁC (HNSW Index): 
                    # Nhận diện khuôn mặt cần độ chính xác cao để tránh False Positive (nhận nhầm người)
                    hnsw_config=HnswConfigDiff(
                        m=32,            # Tăng số lượng liên kết giữa các node (Mặc định: 16)
                        ef_construct=200, # Tăng độ chính xác lúc xây dựng index (Mặc định: 100)
                        on_disk=False     # Giữ index trên RAM để có tốc độ phản hồi (Latency) thấp nhất
                    )
                )
            self.db = db
        except Exception as e:
            print("Error connecting to DB: ", e) 
            self.db = None

class ClientFaces(Faces):
    def __init__(self):
        super().__init__(None)
        self.reload()
        
    def reload(self):
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096) 
        if self.db is None:
            client_path = os.path.join("./vectordb","client") 
            client = QdrantClient(path=client_path)
            self.db = client
        try:
            self.db.get_collection(collection_name=self.collection_name)
            print("Client collection '{}' exists".format(self.collection_name))
        except Exception:
            print("Client collection '{}' not found, creating...".format(self.collection_name))
            self.db.create_collection(
                collection_name= self.collection_name,
                vectors_config= VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                ),
                # # TỐI ƯU RAM & TỐC ĐỘ: Bật Scalar Quantization (Nén về int8)
                # # Giúp giảm 4 lần dung lượng RAM lưu trữ vector nhưng giữ nguyên ~99% độ chính xác
                # quantization_config=QuantizationConfig(
                #     scalar=ScalarQuantization(
                #         type=ScalarType.INT8,
                #         quantile=0.99, # Giữ lại 99% phân phối dữ liệu để tránh mất mát độ chính xác
                #         always_ram=True # Giữ mảng nén trên RAM để tìm kiếm cực nhanh
                #     )
                # ),
                # TỐI ƯU ĐỘ CHÍNH XÁC (HNSW Index): 
                # Nhận diện khuôn mặt cần độ chính xác cao để tránh False Positive (nhận nhầm người)
                hnsw_config=HnswConfigDiff(
                    m=32,            # Tăng số lượng liên kết giữa các node (Mặc định: 16)
                    ef_construct=200, # Tăng độ chính xác lúc xây dựng index (Mặc định: 100)
                    on_disk=False     # Giữ index trên RAM để có tốc độ phản hồi (Latency) thấp nhất
                )
            )
        return self


class ImportFaces(Faces):
    def __init__(self, db = None):
        super().__init__(db)
        self.reload()
        
    def reload(self):
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096) 
        if self.db is None:
            client_path = os.path.join("./vectordb","import") 
            self.db = QdrantClient(path=client_path)
        try:
            self.db.get_collection(collection_name=self.collection_name)
            print("Client Import collection '{}' exists".format(self.collection_name))
        except Exception:
            print("Client Import collection '{}' not found, creating...".format(self.collection_name))
            self.db.create_collection(
                collection_name= self.collection_name,
                vectors_config= VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                ),
            )
        return self
    
    def upsert_face(self, id, vector, payload = {}):
        point = PointStruct(
                        id=id,
                        vector=vector,
                        payload=payload
                    )
        super().upsert_faces([point])
        return self
    
class DbProvider:
    def __init__(self): 
        self.client:ClientFaces = ClientFaces()
        self.import_client:ImportFaces = ImportFaces()
        self.db:RemoteFaces = RemoteFaces()
    
    def close(self): 
        if self.db is not None:
            self.db.close()
        if self.client is not None:
            self.client.close()
        if self.import_client is not None:
            self.import_client.close()
        return self
    
    def reload(self, clear_client=False):
        # Close existing DB connection if any
        if clear_client:
            self.client.clear()
        self.close() 
        self.host = settings.get("VECTORDB","HOST", fallback= "localhost")
        self.port = settings.getint("VECTORDB","PORT", fallback= 6333)
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096)
        self.db.reload()
        self.client.reload()
        self.import_client.reload()
        return self
    
    def get_client(self):
        return self.client
    
    def get_db(self):
        return self.db

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
        db = self.get_db().db
        client = self.get_client().db
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
    
db : DbProvider = DbProvider()