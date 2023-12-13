from threading import Thread
import uuid
import cv2
import numpy as np
import imutils
import time
import os

from deepface import DeepFace
from deepface.commons import functions, distance

from deepface.detectors import FaceDetector

from retinaface import RetinaFace

from PyQt5 import QtCore, QtWidgets, QtGui
from timebounded import TimeBounded

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class FaceRecognize(QtWidgets.QWidget):
    def __init__(self, faces, parent=None, queue_size=1, time_windows=5*60) -> None:
        super(FaceRecognize, self).__init__(parent)     
        self.faces = faces
        self.collection_name="hubt_faces"
        vector_size = 2622
        local_path="./vectordb"
        
        self.client = QdrantClient(path=local_path)

        if not os.path.isfile(local_path + "/collection/hubt_faces/storage.sqlite"):
            self.client.create_collection(
                collection_name= self.collection_name,
                vectors_config= VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        self.face_in_stream = QdrantClient(":memory:")
        self.face_in_stream.create_collection(
            collection_name= self.collection_name,
            vectors_config= VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        self.represents = [] #unique face in videos
        self.recognized =  TimeBounded(maxage=time_windows) #unique face in windows time
        self.recognized_items = [] # face list to show
        self.model_name = "VGG-Face"
        self.time_windows = time_windows
        self.get_frame_thread = Thread(target=self.recognize, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        self.view_widget = QtWidgets.QTableWidget(0, 3)
        self.view_widget.resizeColumnsToContents()
        self.view_widget.resizeRowsToContents()
        # Periodically set video frame to display
        #main thread add item recognize to view
        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_row)
        self.timer.start(2)

    def get_view(self):
        return self.view_widget
    
    def set_row(self): 
        "update UI Widget for recognized items"
        rows = self.view_widget.rowCount()
        recognized = len(self.recognized_items) 
        if rows < recognized:
            print("Add items")
            while rows < recognized:
                detected_face, item_intime, item_recognize = self.recognized_items[rows]
                self.addView(detected_face, item_intime, item_recognize)
                rows = (rows+1)
        else:
            self.spin(1)
    def load_faces(self):
        db = QdrantClient("localhost", port=6333)
        offset = 0

        while offset != None:
            points, offset = db.scroll(
                collection_name=self.collection_name,
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

            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            print(offset)
    #find face by ai
    def recognize(self):
        #Load face data
        self.load_faces()
        DeepFace.build_model(model_name=self.model_name)
        while True:
            try:
                if(len(self.faces) > 0):
                    face = self.faces.pop()
                    img = face["face"]
                    represent = DeepFace.represent(                        
                        img_path=img,
                        model_name=self.model_name,
                        enforce_detection=True, 
                        align=True,
                        detector_backend="skip" #retinaface
                    )
                    
                    #check unique face in stream
                    item = self.find_face_in_stream(represent=represent[0]["embedding"], face_item = face) 

                    if (not self.recognized.exists(item.id)):
                        print("add face from camera to table: ", item.id)
                        #recognize in known db
                        self.recognized.add(item.id, item)
                        item_recognized = self.find(represent=represent[0]["embedding"], face_item = face) 
                        self.recognized_items.append((img, item, item_recognized)) 
            
                        # if(img.max() < 1):
                        #     a = img.copy()
                        #     img = np.interp(a, (a.min(), a.max()), (0, 255)).astype(np.uint8)
                        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
                        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
                        
                        # print("img.shape: ", img.shape)

                        # info = RetinaFace.detect_faces(img_path = img, threshold=0.9) 
                        # if isinstance(info, dict):
                        #     for face_idx in info.keys():
                        #         face["features"] = info[face_idx]
                        #         #check unique face in stream  
                        #         self.recognized.add(item.id, item)
                        #         item_recognized = self.find(represent=represent[0]["embedding"], face_item = face)
                        #         self.recognized_items.append((face["detected"], item, item_recognized))
                        #         break
                        # else:
                        #     print("RetinaFace: Face detect verify fault")

                self.spin(1)
            except Exception as error:
                print("recognize error: ", error)
                self.spin(2)
                pass
    
    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()
    def addView(self, detected_face, item_intime, item_recognize):
        face_icon = item_intime.payload["img"] # imutils.resize(face, width=150)
        pix = None
        if type(face_icon) == str:
            pix =  QtGui.QPixmap(face_icon)
        else:
            pix = face_icon   
        item = QtWidgets.QLabel()
        item.setText("")
        item.setPixmap(pix) 
        item.setScaledContents(True)  

        img = detected_face.copy()# imutils.resize(face, width=150) 
        pix_f = self.convert_cv_qt(img)
        item_f = QtWidgets.QLabel()
        item_f.setText("")
        item_f.setPixmap(pix_f) 
        item_f.setScaledContents(True)  

        item_id = QtWidgets.QLabel()
        if item_recognize == None:
            item_id.setText("Unrecognize")
        else:
            item_id.setText(item_recognize.payload["id"])

        #add item
        rowPosition = self.view_widget.rowCount()
        self.view_widget.insertRow(rowPosition) 
        self.view_widget.setCellWidget(rowPosition, 0, item_f)
        self.view_widget.setCellWidget(rowPosition, 1, item)
        self.view_widget.setCellWidget(rowPosition, 2, item_id)
    def convert_cv_qt(self, cv_img):
        if cv_img.max() < 1:
            a = cv_img.copy()
            cv_img = np.interp(a, (a.min(), a.max()), (0, 255)).astype(np.uint8)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(cv_img.shape[0], cv_img.shape[1], QtCore.Qt.AspectRatioMode.KeepAspectRatio)        
        return QtGui.QPixmap.fromImage(p)
    
    def find_face_in_stream(self, represent, face_item):
        "Find a unique face in video stream"
        data =  self.face_in_stream.search(
            collection_name= self.collection_name, 
            query_vector= represent,
            with_vectors=True,
            with_payload=True,
            limit= 1,
            search_params=models.SearchParams(hnsw_ef=128, exact=True),
            score_threshold=0.7
        )
        # print("search length: ", len(data))
        if len(data) > 0:
            item = data[0] 
            throw = 0.1 # distance.findThreshold(model_name=self.model_name, distance_metric= "cosine")
            d = distance.findCosineDistance(item.vector, represent)  
            if d < throw:                 
                print("face exists in video search score: ", item.score, d, throw)
                return item
        
        #add item if not found
        print("New face in video detected")
        id= uuid.uuid4().urn    
        item = PointStruct(
            id= id,
            vector= represent,
            payload= {
                "class": "undefined",
                "face_region": {"x":face_item["x"], "y":face_item["y"], "w":face_item["w"], "h":face_item["h"]},
                "id": "1",
                "type": "unvalidated",
                "img": self.convert_cv_qt(face_item["detected"])
            },
        )        
        self.face_in_stream.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[item]
        )
        return item
    
    def find(self, face_item, represent):
        data =  self.client.search(
            collection_name= self.collection_name, 
            query_vector= represent,
            with_vectors=True,
            with_payload=True,
            limit= 1,
            search_params=models.SearchParams(hnsw_ef=128, exact=True),
            score_threshold=0.9
        )
        # print("search length: ", len(data))
        if len(data) > 0:
            item = data[0] 
            throw = 0.1 # distance.findThreshold(model_name=self.model_name, distance_metric= "cosine")
            d = distance.findCosineDistance(item.vector, represent)  
            print("search score: ", item.score, d, throw)
            if d < throw: 
                return (item)
        
        #add item if not found
        print("New face unrecognize!")  
        return(None)

