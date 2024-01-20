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

    def __init__(self, width, height, faces, aspect_ratio=False, parent=None, deque_size=1, face_confidence_threshold=0.99):
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

        self.camera_stream_link = None

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QtWidgets.QLabel()
        self.detected_frame = None

        # Start background video source loading
        self.load_video_thread_count = 0
        self.load_video_thread = Thread(target=self.load_network_stream, args=())
        self.load_video_thread.daemon = True
        self.load_video_thread.start()
        
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
        """detect and reconnect network stream if connection is lost"""
        def scan_camera_sources():
            try:
                camera_sources = []
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        camera_sources.append(i)
                        cap.release()
                        break
                if camera_sources:
                    return camera_sources[0]
                else:
                    return None
            except Exception as e:
                print(f"Error occurred while getting camera: {str(e)}")
                return None
            
        def verify_network_stream(link):
            """Attempts to receive a frame from given link"""
            if link is None:
                return False
            cap = cv2.VideoCapture(link)
            if not cap.isOpened():
                return False
            cap.release()
            return True
        
        while True:
            if self.online:
                self.spin(5)
                continue
            self.load_video_thread_count += 1
            self.camera_stream_link = scan_camera_sources()
            if verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
            else:
                print("Camera stream not available.")
                self.spin(1)
                continue

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
            try:
                if self.capture and self.online:
                    # Read next frame from stream and insert into deque
                    if self.capture.isOpened():
                        status, frame = self.capture.read()
                        if status:
                            self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
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
            def create_connecting_image(width, height):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                
                text = "Attempting to connect to IP camera {}".format("." * (self.load_video_thread_count % 4))
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(text, font, 1, 2)
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return image

            connecting_image = create_connecting_image(self.screen_width, self.screen_height)
            img = QtGui.QImage(connecting_image, connecting_image.shape[1], connecting_image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)
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