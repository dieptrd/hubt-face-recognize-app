from collections import deque, OrderedDict
from threading import Thread
import cv2
import numpy as np
import imutils
import time
from deepface import DeepFace
from deepface.commons import functions, distance
from PyQt5 import QtCore, QtWidgets, QtGui
from timebounded import TimeBounded

class FaceRecognize(QtWidgets.QWidget):
    def __init__(self, faces, parent=None, queue_size=1, time_windows=5*60) -> None:
        super(FaceRecognize, self).__init__(parent)     
        self.faces = faces

        self.represents = [] #unique face in videos
        self.recognized =  TimeBounded(maxage=time_windows) #unique face in windows time
        self.recognized_items = [] # face list to show
        self.model_name = "VGG-Face"
        self.time_windows = time_windows
        self.get_frame_thread = Thread(target=self.recognize, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        self.view_widget = QtWidgets.QTableWidget(0, 2)
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
                item = self.recognized_items[rows]
                self.addView(item)
                rows = (rows+1)
        else:
            self.spin(1)
    #find face by ai
    def recognize(self):
        DeepFace.build_model(model_name=self.model_name)
        while True:
            try:
                if(len(self.faces) > 0):
                    face = self.faces.pop()
                    represent = DeepFace.represent(
                        img_path=face["detected"], 
                        model_name=self.model_name,
                        enforce_detection=False, 
                        detector_backend="skip"
                    )
                    status, item = self.find(represent=represent[0]["embedding"], face = face)

                    if status >= 0 and (not self.recognized.exists(status)):
                        print("Unique face from camera")
                        item['recognized_time'] = time.time()
                        item['index'] = status
                        self.recognized.add(status, item) 
                        self.recognized_items.append(item)
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
    def addView(self, face): 
        face_icon = face["detected"].copy()# imutils.resize(face, width=150)
        pix = self.convert_cv_qt(face_icon)        
        item = QtWidgets.QLabel()
        item.setText("")
        item.setPixmap(pix) 
        item.setScaledContents(True)  

        face_detect = face["face"].copy()# imutils.resize(face, width=150)
        a  = cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB)
        img = np.interp(a, (a.min(), a.max()), (0, 255)).astype(np.uint8)
        pix_f = self.convert_cv_qt(img)
        item_f = QtWidgets.QLabel()
        item_f.setText("")
        item_f.setPixmap(pix_f) 
        item_f.setScaledContents(True)  

        #add item
        rowPosition = self.view_widget.rowCount()
        self.view_widget.insertRow(rowPosition) 
        self.view_widget.setCellWidget(rowPosition, 0, item)
        self.view_widget.setCellWidget(rowPosition, 1, item_f)
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(cv_img.shape[0], cv_img.shape[1], QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)
    def find(self, face, represent):
        throw = distance.findThreshold(model_name=self.model_name, distance_metric= "cosine")/2
        for index, item in enumerate(self.represents): 
            d = distance.findCosineDistance(item["represent"], represent) 
            if d < throw: 
                return (index, item.copy())

        #add item if not found
        item = face.copy()
        item["represent"] = represent
        self.represents.append(item)
        return (-1, item) 

