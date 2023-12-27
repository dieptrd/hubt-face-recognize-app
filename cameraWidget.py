from PyQt5 import QtCore, QtWidgets, QtGui 
from threading import Thread
from collections import deque
from datetime import datetime
import time
import cv2
from deepface import DeepFace
from deepface.commons import functions
import imutils
import numpy as np
from appSettings import settings
class CameraWidget(QtWidgets.QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd, dlib, mediapipe or yolov8.
    """

    def __init__(self, width, height, faces, stream_link=0, aspect_ratio=False, parent=None, deque_size=1, face_confidence_threshold=0.99):
        super(CameraWidget, self).__init__(parent)
        
        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)
        self.faces = faces
        self.face_confidence = face_confidence_threshold
        self.face_last = deque(maxlen=2)
        # Slight offset is needed since PyQt layouts have a built in padding
        # So add offset to counter the padding 
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        self.camera_stream_link = stream_link

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QtWidgets.QLabel()
        self.detected_frame = None

        self.load_network_stream()
        self.update_recognize()

        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        #start background face detect
        self.detect_face_thread = Thread(target=self.detect_face, args=())
        self.detect_face_thread.daemon = True
        self.detect_face_thread.start()

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(2)

        print('Started camera: {}'.format(self.camera_stream_link))

    def load_network_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def verify_network_stream(self, link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
            try:
                if self.capture.isOpened() and self.online:
                    # Read next frame from stream and insert into deque
                    status, frame = self.capture.read()
                    if status:
                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    # Attempt to reconnect
                    print('attempting to reconnect', self.camera_stream_link)
                    self.load_network_stream()
                    self.spin(2)
            except AttributeError:
                pass
    
    def update_recognize(self):
        self.detector_backend = settings.get("PROCESSING", "DETECTED_METHOD", fallback="retinaface")
        self.model_name = settings.get("PROCESSING", "RECOGNIZE_METHOD", fallback="VGG-Face")
        self.wait_recognize = settings.get("PROCESSING", "WAIT_RECOGNIZED", fallback="True") == "True"
        print(f'detector: {self.detector_backend}, model: {self.model_name}, slow: {self.wait_recognize}')
        DeepFace.build_model(model_name= self.model_name)
        self.target_size = functions.find_target_size(model_name= self.model_name)

    def detect_face(self):
        """get face from frame""" 
        # build models once to store them in the memory
        # otherwise, they will be built after cam started and this will cause delays 
        while True:            
            if self.wait_recognize and len(self.faces) >0: 
                self.spin(1)
                continue
            if len(self.deque) < 1:
                self.spin(2)
                continue
            try:
                frame = (self.deque[-1]).copy()
                face_objs = DeepFace.extract_faces(
                    img_path=frame.copy(),
                    target_size=self.target_size,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=False
                ) 
                w_bigger = 0
                face_bigger = None
                for face_obj in face_objs:
                    facial_area = face_obj["facial_area"]
                    if facial_area["w"] > 50 and face_obj["confidence"] > self.face_confidence:
                        if w_bigger < facial_area["w"]:
                            w_bigger = facial_area["w"] 
                            face_bigger = facial_area.copy()
                if w_bigger > 0:
                    item = face_bigger.copy()
                    item['frame'] = frame.copy()
                    item["tic"] = time.time()
                    self.faces.append(item.copy())
                    self.face_last.append(item.copy()) 
            except Exception as e:
                print("Detect face e: ", e)
                self.spin(1)
                pass 
    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""
        if not self.online:
            self.spin(1)
            return

        if self.deque and self.online:
            # Grab latest frame
            frame = (self.deque[-1]).copy()
            if len(self.face_last) > 0:
                face = self.face_last[-1]
                if time.time() - face['tic'] < 5:
                    x = face["x"]
                    y = face["y"]
                    w = face["w"]
                    h = face["h"]
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (67, 67, 67), 1
                    )  # draw rectangle to main image
            # Keep frame aspect ratio
            if self.maintain_aspect_ratio:
                self.frame = imutils.resize(frame, width=self.screen_width)
            # Force resize
            else:
                self.frame = cv2.resize(frame, (self.screen_width, self.screen_height))

            # Add timestamp to cameras
            cv2.rectangle(self.frame, (self.screen_width-190,0), (self.screen_width,50), color=(0,0,0), thickness=-1)
            cv2.putText(self.frame, datetime.now().strftime('%H:%M:%S'), (self.screen_width-185,37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), lineType=cv2.LINE_AA)

            # Convert to pixmap and set to video frame
            img = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)

        if len(self.face_last) > 0 and self.detected_frame is not None:
            face = (self.face_last[-1]).copy()
            if not hasattr(self, "detected_frame_tic"):
                self.detected_frame_tic = 0
            if self.detected_frame_tic != face["tic"]:
                self.detected_frame_tic = face["tic"]
                x,y,w,h = face["x"], face["y"], face["w"], face["h"]
                img_face = face["frame"][y : y + h, x : x + w]  # crop detected face
                img_face = imutils.resize(img_face, width=200)
                img_face = QtGui.QImage(img_face, img_face.shape[1], img_face.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                pix_face = QtGui.QPixmap.fromImage(img_face)
                self.detected_frame.setPixmap(pix_face)

    def get_video_frame(self):
        return self.video_frame
    def get_face_detected_frame(self):        
        if self.detected_frame is None:
            self.detected_frame = QtWidgets.QLabel()
        return self.detected_frame