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

from logger import logger
from importDialog import ImportDialog
from timebounded import TimeBounded

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings
from faceCompareWidget import FaceCompareWidget
from db import db
import commons

class FaceRecognize(QtWidgets.QWidget):
    def __init__(self, faces, face_recognized, face_new=None, parent=None ) -> None:
        super(FaceRecognize, self).__init__(parent)     
        self.faces = faces
        self.face_recognized = face_recognized
        self.face_new = face_new
        
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096) 
        self.model_name = settings.get("PROCESSING", "recognize_method", fallback="VGG-Face")

        self.recognize_thread = None
        # self.reload_recognize_thread()

        self.text_log = deque(maxlen=5)
        self.recognize_frame_queue = deque(maxlen=3)
        self.recognize_frame = None

        self.view_widget = None

        # Periodically set video frame to display
        #main thread add item recognize to view
        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ui_update)
        self.timer.start(2)

    def ui_update(self):
        while self.text_log:
            if self.view_widget is not None:
                text = self.text_log.pop()
                if isinstance(text, str):
                    self.view_widget.append(text)
                else:
                    #add image to log
                    (image, text) = text
                    self.view_widget.insertHtml("<img src='{}'> {}".format(self.convert_image_base64(image), text)) 
        
                self.view_widget.verticalScrollBar().setValue(self.view_widget.verticalScrollBar().maximum())
            
        while self.recognize_frame_queue:
            (face_on_cam, registed_item) = self.recognize_frame_queue.pop()
            if self.recognize_frame is not None:
                img = imutils.resize(face_on_cam, width=200)
                pix_face = self.convert_cv_qt(img, "Face Recognized" if registed_item is not None else "New face")
                msv = commons._safe_get(registed_item, "payload", "msv", default="") if registed_item is not None else ""
                name = commons._safe_get(registed_item, "payload", "fullname", default="") if registed_item is not None else ""
                
                self.recognize_frame.set_camera_face(pix_face)
                self.recognize_frame.set_info(msv=msv, name=name)

    def get_new_faces_view(self):
        if self.view_widget is None:
            self.view_widget = QtWidgets.QTextEdit(self)
            self.view_widget.setReadOnly(True)
        return self.view_widget
    
    def get_recognize_frame(self, show_info=False):
        if self.recognize_frame is None:
            self.recognize_frame = FaceCompareWidget(self, show_info=show_info)
        return self.recognize_frame

    def reload_recognize_thread(self):
        # Wait last thread stop before start new thread
        if self.recognize_thread:
            while(self.recognize_thread.is_alive()):
                self.recognize_thread_wait_stop = True
                commons.spin(0.5)
        
        self.recognize_thread_wait_stop = False
        
        #reload setting
        self.collection_name = settings.get("VECTORDB", "COLLECTION_NAME", fallback="hubt_faces")
        self.vector_size = settings.getint("VECTORDB", "VECTOR_SIZE", fallback= 4096)
        self.model_name = settings.get("PROCESSING", "RECOGNIZE_METHOD", fallback="VGG-Face")
        
        #load all faces from db to local client
        self.recognize_thread = Thread(target=self.recognize, args=())
        self.recognize_thread.daemon = True
        self.recognize_thread_wait_stop = False
        self.recognize_thread.start()

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
                    # logger.info("Face recognition time: %s, face_confidence: %s", toc, represent[0].get("face_confidence") if represent is not None and len(represent) > 0 else 0)
                    print("Face recognition time: ", toc, " face_confidence: ", represent[0].get("face_confidence") if represent is not None and len(represent) > 0 else 0)
                    item_in_db = None
                    #check unique face in local db
                    if(represent is not None and len(represent) > 0):
                        item_in_db = self.find_face_in_db(represent= represent[0].get("embedding"))
                        if item_in_db is not None:
                            recognized_item = face.copy()
                            recognized_item["recognized"] = item_in_db
                            
                            self.recognize_frame_queue.append((face_mark, item_in_db))
                            if self.face_recognized is not None:
                                self.face_recognized.append(recognized_item)
                        else:
                            id = face.get("id", str(uuid.uuid4()))
                            # self.text_log.append("Face detected not found in local db.") 
                            self.text_log.append((face_mark, "add new face to db: " + id))
                            payload= {
                                "face_area": face_area,
                                "face": self.convert_image_base64(face_mark),
                                "frame": self.convert_image_base64(frame),
                            }
                            
                            if self.face_new is not None: 
                                self.face_new.append((id,represent[0].get("embedding"), payload))
                            self.recognize_frame_queue.append((face_mark, None))
            except Exception as error:
                print("recognize error: ", error) 
                # self.spin(2)
                pass
    
    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()
    
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
        resp =  db.get_client().query_points(
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

        data =  db.get_client().retrieve(
            collection_name= self.collection_name,
            ids= [id],
            with_vectors=True,
            with_payload=True
        )
        return data[0] if len(data) > 0 else None
    
    def add_face_to_db(self, item):
        db.get_client().upsert(
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

