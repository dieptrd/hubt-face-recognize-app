from threading import Thread
import uuid
import cv2
import numpy as np
import imutils
import time
import os

from collections import deque
from deepface import DeepFace
from deepface.commons import functions, distance

from deepface.detectors import FaceDetector

from retinaface import RetinaFace

from PyQt5 import QtCore, QtWidgets, QtGui
from timebounded import TimeBounded

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings

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

        self.recognize_thread = Thread(target=self.recognize, args=())
        self.recognize_thread.daemon = True
        self.recognize_thread_wait_stop = False
        self.reload_recognize_thread()

        self.recognize_frame_queue = deque(maxlen=1)
        self.recognize_frame = None

        self.view_widget = QtWidgets.QTableWidget(0, 3)
        self.view_widget.setHorizontalHeaderLabels(["Detected", "In Stream", "In Database"])
        self.view_widget.resizeColumnsToContents()
        self.view_widget.resizeRowsToContents()
        self.view_widget.doubleClicked.connect(self.import_item)

        # Periodically set video frame to display
        #main thread add item recognize to view
        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_row)
        self.timer.start(2)

    def get_view(self):
        return self.view_widget
    
    def get_recognize_frame(self):
        if self.recognize_frame is None:
            self.recognize_frame = QtWidgets.QLabel()
        return self.recognize_frame

    def set_row(self): 
        "update UI Widget for recognized items"
        # update recognize items view
        rows = self.view_widget.rowCount()
        recognized = len(self.recognized_items) 
        if rows < recognized:
            # print("Add items")
            while rows < recognized:
                detected_face, item_instream, item_recognize = self.recognized_items[rows]
                self.addView(detected_face, item_instream, item_recognize)
                rows = (rows+1)
        # update recognize frame view
        if len(self.recognize_frame_queue) > 0 and self.recognize_frame is not None:
            img = self.recognize_frame_queue.pop()
            img = imutils.resize(img, width=200)
            pix_face = self.convert_cv_qt(img)
            self.recognize_frame.setPixmap(pix_face)
    
    def reload_recognize_thread(self):
        if self.recognize_thread:
            while(self.recognize_thread.is_alive()):
                self.recognize_thread_wait_stop = True
                self.spin(1)
        
        self.recognize_thread_wait_stop = False
        self.load_faces()
        DeepFace.build_model(model_name=self.model_name)
        self.recognize_thread.start()
        
    def load_faces(self):
        try:
            #wait recognize thread close
            
            host = settings.get("VECTORDB","HOST", fallback= "localhost")
            port = settings.getint("VECTORDB","PORT", fallback= 6333)
            db = QdrantClient(host, port=port)
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

                info = self.client.get_collection(collection_name=self.collection_name)
                print("load faces pages: ", info.points_count)
        except Exception as e:
            print("load_faces error: ", e)
            pass
    #find face by ai
    def recognize(self):        
        while True:
            if self.recognize_thread_wait_stop:
                break 
            try:
                if(len(self.faces) > 0):
                    #crop full face with background
                    face = self.faces.pop()
                    frame = face['frame']
                    # crop detected face with some outsize
                    x,y,w,h = face["x"], face["y"], face["w"], face["h"]
                    crop = frame[y-int(h/2) : y + h + int(y/2), x-(int(x/4)) : x + w + int(x/2)]  

                    #refind face in frame with retinaface methods
                    face_objs = DeepFace.extract_faces(
                        img_path= crop.copy(),
                        target_size= functions.find_target_size(self.model_name),
                        enforce_detection= True, 
                        align= True,
                        detector_backend= "retinaface", #skip
                    )

                    #get bigger face
                    face_width = 0
                    face_bigger = None
                    face_mark = None
                    for face_obj in face_objs:
                        facial_area = face_obj["facial_area"]
                        if facial_area["w"] > 50 and face_obj["confidence"] > 0.99:
                            if face_width <  facial_area["w"]:
                                face_width =  facial_area["w"]
                                face_bigger = facial_area
                                face_mark = face_obj["face"]
                    
                    if face_bigger is None:
                        raise Exception("Face not found.")
                    
                    #crop face
                    x,y,w,h = face_bigger["x"], face_bigger["y"], face_bigger["w"], face_bigger["h"]
                    face_bigger['detected'] = crop[y : y + h, x : x + w]
                    face_bigger['face'] = face_mark.copy()

                    # save face to show on UI
                    self.recognize_frame_queue.append(face_mark.copy()) 

                    #convert finded face to vector
                    represent = DeepFace.represent(                        
                        img_path= face_mark.copy(),
                        model_name= self.model_name,
                        enforce_detection= False, 
                        align= True,
                        normalization="VGGFace",
                        detector_backend= "skip"
                    ) 
                    
                    #check unique face in stream
                    item = self.find_face_in_stream(represent=represent[0]["embedding"], face_item = face_bigger) 
                     
                    if (not self.recognized.exists(item.id)):
                        print("add face from camera to table: ", item.id)
                        #recognize in known db
                        self.recognized.add(item.id, item)
                        item_recognized = self.find_face_in_db(represent=represent[0]["embedding"])
                        self.recognized_items.append((face_bigger['detected'].copy(), item, item_recognized))
                        print("3") 
            except Exception as error:
                print("recognize error: ", type(error).__module__, error)
                self.spin(2)
                pass
    def _rever_image(self, img):
        if(img.max() < 1):
            a = img.copy()
            img = np.interp(a, (a.min(), a.max()), (0,255)).astype(np.uint8) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
        return img
    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()
    def addView(self, detected_face, item_instream, item_recognize):
        try:
            face_icon, _ = functions.load_image(item_instream.payload["img"]) # imutils.resize(face, width=150)
        
            item = QtWidgets.QLabel("", self)
            item.setPixmap(self.convert_cv_qt(face_icon)) 
            item.setScaledContents(True)  
            
            item_f = QtWidgets.QLabel("", self) 
            item_f.setPixmap(self.convert_cv_qt(detected_face)) 
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
        except Exception as ex:
            print("add row error: ", ex)
            pass
    def import_item(self, item):
        print(f"Item {item.row()}, {item.column()} was double-clicked")
        detected_face, item_instream, item_recognize = self.recognized_items[item.row()]

    def _rever_image(self, img):
        if(img.max() < 1):
            a = img.copy()
            img = np.interp(a, (a.min(), a.max()), (0,255)).astype(np.uint8) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
        return img
    
    def convert_cv_qt(self, img):
        cv_img = self._rever_image(img)

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
            throw = distance.findThreshold(model_name=self.model_name, distance_metric= "cosine")
            d = distance.findCosineDistance(item.vector, represent)  
            if d < throw:                 
                # print("face exists in video search score: ", item.score, d, throw)
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
                "img": face_item["detected"]
            },
        )        
        self.face_in_stream.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[item]
        )
        return item
    
    def find_face_in_db(self, represent):
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
            # print("search score: ", item.score, d, throw)
            if d < throw: 
                return (item)
        
        #add item if not found
        print("New face unrecognize!")  
        return(None)

