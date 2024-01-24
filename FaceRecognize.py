from threading import Thread
import uuid
import cv2
import numpy as np
import imutils
import time
import os
import base64

from collections import deque
from deepface import DeepFace
from deepface.commons import functions, distance

from PyQt5 import QtCore, QtWidgets, QtGui
from importDialog import ImportDialog
from timebounded import TimeBounded

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings


class FaceRecognize(QtWidgets.QWidget):
    def __init__(self, faces, parent=None, queue_size=1) -> None:
        super(FaceRecognize, self).__init__(parent)     
        self.faces = faces
        self.collection_name="hubt_faces"
        self.vector_size = 2622
        local_path="./vectordb"
        time_windows = settings.getint("PROCESSING", "TIME_WINDOWS", fallback=5*60)

        client_path = os.path.join(local_path,"client")
        self.client = QdrantClient(path=client_path)

        if not os.path.isfile(client_path + "/collection/{}/storage.sqlite".format(self.collection_name)):
            self.client.create_collection(
                collection_name= self.collection_name,
                vectors_config= VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        
        # self.face_in_stream = QdrantClient(":memory:")
        in_stream_path = os.path.join(local_path,"in_stream")
        self.face_in_stream = QdrantClient(path=in_stream_path)
        self.face_in_stream.recreate_collection(
            collection_name= self.collection_name,
            vectors_config= VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

        self.recognized =  TimeBounded(maxage=time_windows) #unique face in windows time
        self.recognized_items = [] # face list to show
        self.model_name = "VGG-Face"
        

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
                detected,_in_stream_id, _in_database_id = self.recognized_items[rows]
                item_instream = self.get_face_in_stream(_in_stream_id)
                item_indatabase = self.get_face_in_db(_in_database_id)
                self.addView(detected, item_instream, item_indatabase)
                rows = (rows+1)
        # update recognize frame view
        if len(self.recognize_frame_queue) > 0 and self.recognize_frame is not None:
            img = self.recognize_frame_queue.pop()
            img = imutils.resize(img, width=200)
            pix_face = self.convert_cv_qt(img, text="Face Recognized")
            self.recognize_frame.setPixmap(pix_face)
    
    def reload_recognize_thread(self):
        if self.recognize_thread:
            while(self.recognize_thread.is_alive()):
                self.recognize_thread_wait_stop = True
                self.spin(0.5)
        
        self.recognize_thread_wait_stop = False
        self.progress_dialog = QtWidgets.QProgressDialog()
        self.progress_dialog.setLabelText("DB Loading...")
        self.progress_dialog.setRange(0, 1000)
        self.progress_dialog.setModal(True)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.show()

        self.client.recreate_collection(
            collection_name= self.collection_name,
            vectors_config= VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        DeepFace.build_model(model_name=self.model_name)
        self.load_faces()        
        self.recognize_thread.start() 

    def load_faces(self):        
        try:
            host = settings.get("VECTORDB","HOST", fallback= "localhost")
            port = settings.getint("VECTORDB","PORT", fallback= 6333)
            db = QdrantClient(host, port=port)
            offset = 0
            #wait recognize thread close
            matchs = settings.class_name()
            matchs.append("undefined")
            
            while offset != None:
                points, offset = db.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="class",
                                match=models.MatchAny(any=matchs),
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
                self.progress_dialog.setValue(info.points_count)
                print("load faces pages: ", info.points_count)
        except Exception as e:
            print("load_faces error: ", e)
            pass
        finally:
            if db is not None:
                db.close()
            self.progress_dialog.close()

    def insert_new_face(self, item):
        item_insert = PointStruct(
            id= item.id,
            vector= item.vector,
            payload= item.payload,
        )    
        #insert to db server
        host = settings.get("VECTORDB","HOST", fallback= "localhost")
        port = settings.getint("VECTORDB","PORT", fallback= 6333)
        db = QdrantClient(host, port=port)
        db.upsert(
            collection_name=self.collection_name,
            points= [item_insert],
            wait=True
        ) 
        db.close()
        #insert to db local
        self.client.upsert(
            collection_name=self.collection_name,
            points= [item_insert],
            wait=True
        )

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
                    face_bigger['frame'] = crop
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
                        print("detect a new face on camera stream: ", item.id) 
                        self.recognized.add(item.id, item)
                        
                        #check unique face in client db
                        item_in_db = self.find_face_in_db(represent= item.vector)
                        if item_in_db is None:
                            item_in_db = self.find_face_in_db(represent=represent[0]["embedding"])
                        
                        recognized_item = (face_bigger['detected'], item.id, item_in_db.id if item_in_db else None)
                        self.recognized_items.append(recognized_item)
                        print("recognized_items id: ", item.id)
            except Exception as error:
                print("recognize error: ", error)
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
            item.setFixedSize(80, 80)
            item.setScaledContents(True)  

            item_f = QtWidgets.QLabel("", self) 
            item_f.setPixmap(self.convert_cv_qt(detected_face)) 
            item_f.setFixedSize(80, 80)
            item_f.setScaledContents(True)  

            item_id = QtWidgets.QLabel()
            if item_recognize == None:
                item_id.setText("Unrecognize")
            else:
                item_id.setText(item_recognize.payload["id"])

            #add item
            rowPosition = 0  # Set rowPosition to 0 to insert row at the top
            self.view_widget.insertRow(rowPosition) 
            self.view_widget.setRowHeight(rowPosition, 80)

            self.view_widget.setColumnWidth(0, 80)
            self.view_widget.setCellWidget(rowPosition, 0, item_f)  

            self.view_widget.setColumnWidth(1, 80)          
            self.view_widget.setCellWidget(rowPosition, 1, item)

            self.view_widget.setCellWidget(rowPosition, 2, item_id)
        except Exception as ex:
            print("add row error: ", ex)
            pass
    def import_item(self, item):
        def update_item_view(item):
            item_id = item.id
            for rowPosition, rowItem in enumerate(self.recognized_items):
                detected, current_item_instream_id, in_db_item_id = rowItem
                if current_item_instream_id == item_id and in_db_item_id is None:
                    self.recognized_items[rowPosition] = (detected, current_item_instream_id, current_item_instream_id)
                    self.view_widget.setCellWidget(rowPosition, 2, QtWidgets.QLabel(text=str(item.payload["name"])))

            
        row_position = item.row()
        _, item_instream_id , item_recognize = self.recognized_items[row_position]
        if item_recognize is None:
            dlg = ImportDialog(self)
            result = dlg.exec()
            if result and len(dlg.studentId.text()) > 0 and len(dlg.className.text()) > 0:
                item_instream = self.get_face_in_stream(item_instream_id)
                item_instream.payload["id"] = dlg.studentId.text()
                item_instream.payload["name"] = dlg.studentName.text()
                item_instream.payload["class"] = dlg.className.text()

                self.insert_new_face(item_instream)
                update_item_view(item_instream)
                print("Add new student success: ", dlg.studentId.text(), item_instream.id)

    def _rever_image(self, img):
        if(img.max() < 1):
            a = img.copy()
            img = np.interp(a, (a.min(), a.max()), (0,255)).astype(np.uint8) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
        return img
    
    def convert_cv_qt(self, img, text=None):
        cv_img = self._rever_image(img)

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        if text is not None:            
            cv2.putText(rgb_image, text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)
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
                "id": "1",
                "name": "undefined",
                "type": "unvalidated",
                "face_region": {"x":face_item["x"], "y":face_item["y"], "w":face_item["w"], "h":face_item["h"]},
                "face": self.convert_image_base64(face_item["face"]),
                "detected": self.convert_image_base64(face_item["detected"]),
                "img": self.convert_image_base64(face_item["frame"])
            },
        )    
            
        self.face_in_stream.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[item]
        )
        return item
    
    def get_face_in_stream(self, id):
        data =  self.face_in_stream.retrieve(
            collection_name= self.collection_name, 
            ids= [id],
            with_vectors=True,
            with_payload=True
        )
        return data[0] if len(data) > 0 else None
    def convert_image_base64(self, img, ext= ".jpg"):
        a, _ = functions.load_image(img)
        img_base64 = base64.b64encode(cv2.imencode(ext, img)[1]).decode()
        return f'data:image/{ext};base64,' + img_base64
    
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
    
    def get_face_in_db(self, id):
        if id is None:
            return None
        
        data =  self.client.retrieve(
            collection_name= self.collection_name, 
            ids= [id],
            with_vectors=True,
            with_payload=True
        )
        return data[0] if len(data) > 0 else None
    
    def add_face_in_db(self, item):
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[item]
        )

