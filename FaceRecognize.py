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
from deepface.modules import verification

from PyQt5 import QtCore, QtWidgets, QtGui
from importDialog import ImportDialog
from timebounded import TimeBounded

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings
from faceCompareWidget import FaceCompareWidget
from logger import logger
from db import DbProvider

class FaceRecognize(QtWidgets.QWidget):
    def __init__(self, faces, face_recognized, parent=None ) -> None:
        super(FaceRecognize, self).__init__(parent)     
        self.faces = faces
        self.face_recognized = face_recognized
        self.db = DbProvider()
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096)
        time_windows = settings.getint("PROCESSING", "TIME_WINDOWS", fallback=5*60)

        client_path = os.path.join("./vectordb","client")
        
        self.client = QdrantClient(path=client_path)
        if not os.path.isfile(client_path + "/collection/{}/storage.sqlite".format(self.collection_name)):
            self.client.create_collection(
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

        self.text_log = deque(maxlen=10)
        self.recognize_frame_queue = deque(maxlen=10)
        self.recognize_frame = None

        self.view_widget = QtWidgets.QTextEdit()
        self.view_widget.setReadOnly(True)

        # Periodically set video frame to display
        #main thread add item recognize to view
        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ui_refresh)
        self.timer.start(2)

    def ui_refresh(self):
        while self.text_log:
            text = self.text_log.popleft()
            if isinstance(text, str):
                self.view_widget.append(text)
            else:
                #add image to log
                (image, text) = text
                self.view_widget.insertHtml("<img src='{}'> {}".format(self.convert_image_base64(image), text)) 
    
            self.view_widget.verticalScrollBar().setValue(self.view_widget.verticalScrollBar().maximum())
            
        while self.recognize_frame_queue:
            (face_on_cam, text) = self.recognize_frame_queue.pop()
            if self.recognize_frame is not None:
                img = imutils.resize(face_on_cam, width=200)
                pix_face = self.convert_cv_qt(img, text=text or "Face Recognized")
                self.recognize_frame.set_camera_face(pix_face)

    def get_view(self):
        return self.view_widget
    
    def get_recognize_frame(self):
        if self.recognize_frame is None:
            self.recognize_frame = FaceCompareWidget(self)
        return self.recognize_frame

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
            # Ensure remote collection exists; create it if missing
            try:
                db.get_collection(collection_name=self.collection_name)
                print("Remote collection '{}' exists".format(self.collection_name))
            except Exception:
                print("Remote collection '{}' not found, creating...".format(self.collection_name))
                db.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
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
                    # print("Start recognize face...")
                    #crop full face with background
                    face = self.faces.pop()

                    frame = face['frame']
                    # crop detected face with some outsize
                    face_mark = face["face_crop"].copy()
                    face_area = face["facial_area"]
                    x,y,w,h = face_area["x"], face_area["y"], face_area["w"], face_area["h"]
                    crop = frame[y-int(h/2) : y + h + int(y/2), x-(int(x/4)) : x + w + int(x/2)]   
                    
                    #convert finded face to vector
                    tic = time.time()
                    represent = DeepFace.represent(                        
                        img_path= face_mark,
                        model_name= self.model_name,
                        enforce_detection= False, 
                        align= True,
                        return_face = False,
                        normalization="VGGFace",
                        detector_backend= "skip"
                    ) 
                    toc = time.time() - tic
                    logger.info("Face recognition time: %s, face_confidence: %s", toc, represent[0].get("face_confidence") if represent is not None and len(represent) > 0 else 0)
                    
                    item_in_db = None
                    #check unique face in local db
                    if(represent is not None and len(represent) > 0):
                        item_in_db = self.find_face_in_db(represent= represent[0].get("embedding"))
                        if item_in_db is not None:
                            self.recognize_frame_queue.append((face_mark, "registered: " + item_in_db.id))
                            recognized_item = face.copy()
                            recognized_item["recognized"] = item_in_db
                            self.face_recognized.append(recognized_item)
                        else:
                            # self.text_log.append("Face detected not found in local db.") 
                            id = str(uuid.uuid4())
                            self.text_log.append((face_mark, "add new face to db: " + id))
                            self.add_face_to_db(PointStruct(
                                id= id,
                                vector= represent[0].get("embedding"),
                                payload= {
                                    "msv": None,
                                    "name": None,
                                    "class": None,
                                }
                            ))
                            self.recognize_frame_queue.append((face_mark, "New face"))
            except Exception as error:
                print("recognize error: ", error) 
                # self.spin(2)
                pass
    
    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()
            
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

    def convert_image_base64(self, img, size=100, ext=".jpg"):
        resize_img = imutils.resize(img, width=size)
        resize_img = self._rever_image(resize_img)
        img_base64 = base64.b64encode(cv2.imencode(ext, resize_img)[1]).decode()
        return f'data:image/{ext};base64,' + img_base64
    
    def find_face_in_db(self, represent):
        resp =  self.client.query_points(
            collection_name= self.collection_name, 
            query= represent,
            with_vectors=True,
            with_payload=True,
            limit= 5,
            search_params=models.SearchParams(hnsw_ef=128, exact=True),
            score_threshold=0.8
        )
        
        data = self._normalize_qdrant_response(resp)   

        min_item = min(data, key=lambda x: verification.find_distance(x.vector, represent, "cosine")) if len(data) > 0 else None
        # print("find_face_in_db: ", min_item.id, min_item.score) if min_item is not None else print("find_face_in_db: no item found" )
        # print ("find_face_in_db: ", len(data), " min distance: ", verification.find_distance(min_item.vector, represent, "cosine") if min_item is not None else None)   
        
        # if min_item is not None and verification.find_distance(min_item.vector, represent, "cosine") < throw:
        #     return min_item

        #add item if not found
        if min_item is None:
            print("New face unrecognize!")  
        else:
            print("Face recognized in db: ", min_item.id, " distance: ", verification.find_distance(min_item.vector, represent, "cosine"))
        return min_item
    
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
    
    def add_face_to_db(self, item):
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[item]
        )

    def _normalize_qdrant_response(self, resp):
        """
        Normalize qdrant-client responses to a plain list of results.
        Handles: list, QueryResponse/SearchResult (with .result or .matches), older variants.
        """
        if resp is None:
            return []
        #check status is ok
        if hasattr(resp, "status") and resp.status != "ok":
            print("Qdrant response status not ok: ", resp.status)
            return []
        # already a list
        if isinstance(resp, list):
            return resp
        if hasattr(resp, "points"):
            return resp.points if isinstance(resp.points, list) else list(resp.points)
        # new-style objects
        if hasattr(resp, "result"):
            try:
                return list(resp.result.points) if hasattr(resp.result, "points") else list(resp.result)
            except Exception:
                pass
        if hasattr(resp, "matches"):
            try:
                return list(resp.matches)
            except Exception:
                pass
        # fallback: try to iterate
        try:
            return list(resp)
        except Exception:
            return []

