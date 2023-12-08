from PyQt5 import QtCore, QtWidgets, QtGui 
from threading import Thread
from collections import deque
from datetime import datetime
import time
import cv2
from deepface import DeepFace
from deepface.commons import functions
import imutils
class CameraWidget(QtWidgets.QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into fraame
    """

    def __init__(self, width, height, faces, stream_link=0, aspect_ratio=False, parent=None, deque_size=1):
        super(CameraWidget, self).__init__(parent)
        
        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)
        self.faces = faces
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

        self.load_network_stream()
        
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
    def detect_face(self):
        """get face from frame"""
        model_name = "VGG-Face"
        detector_backend = "opencv"
        # build models once to store them in the memory
        # otherwise, they will be built after cam started and this will cause delays
        DeepFace.build_model(model_name=model_name)
        target_size = functions.find_target_size(model_name=model_name)
        while True:
            if len(self.deque) > 0:                
                try:
                    frame = self.deque[-1]
                    face_objs = DeepFace.extract_faces(
                        img_path=frame,
                        target_size=target_size,
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        align=True
                    ) 
                    w_bigger = 0
                    face_bigger = None
                    for face_obj in face_objs:
                        facial_area = face_obj["facial_area"]
                        if facial_area["w"] > 50 and face_obj["confidence"] > 0.99:
                            if w_bigger < facial_area["w"]:
                                w_bigger = facial_area["w"] 
                                face_bigger = facial_area.copy()
                                face_bigger["face"] = face_obj["face"]
                    if w_bigger > 0:
                        x = face_bigger["x"]
                        y = face_bigger["y"]
                        w = face_bigger["w"]
                        h = face_bigger["h"]
                        detected_face = frame[y : y + h, x : x + w]  # crop detected face
                        item = face_bigger.copy()                        
                        item['detected'] = detected_face
                        self.add_face(item)
                except Exception as e:
                    print("Detect face e: ", e)
                    self.spin(1)
                    pass
            else:
                self.spin(2)

    def add_face(self, face_item):        
        face_item["tic"] = time.time()
        self.faces.append(face_item)
        self.face_last.append(face_item)
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
            frame = self.deque[-1]
            if len(self.face_last) > 0:
                face = self.face_last[-1]
                if time.time() - face['tic'] < 2:
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

    def get_video_frame(self):
        return self.video_frame