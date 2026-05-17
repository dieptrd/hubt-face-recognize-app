import uuid
from PyQt5 import QtCore, QtWidgets, QtGui 
from threading import Thread
from collections import deque
from datetime import datetime
import time
import cv2
from deepface import DeepFace
import importlib
import imutils
import numpy as np
from appSettings import settings
import commons
from logger import logger

class CameraWidget(QtWidgets.QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd, dlib, mediapipe or yolov8.
    """

    def __init__(self, width, height, faces, face_recognized, aspect_ratio=False, parent=None, fps=16, face_tracking=None, deque_size=1, face_confidence_threshold=0.7):
        super(CameraWidget, self).__init__(parent)
        
        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)
        self.faces = faces
        self.face_recognized = face_recognized
        self.last_face = deque(maxlen=1)
        self.face_confidence = face_confidence_threshold
        self.fps = fps
        self.face_tracking = face_tracking
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

        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        #start background face detecting
        self.detect_face_thread = None
        # self.update_recognize()

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(2)

    def load_network_stream(self):
        """detect and reconnect network stream if connection is lost"""
        def scan_camera_sources():
            try:
                camera_sources = []
                for i in range(10):
                    self.load_video_thread_count += 1
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
            self.load_video_thread_count += 1
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
                commons.spin(5)
                continue
            self.camera_stream_link = scan_camera_sources()
            if verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
            else:
                print("Camera stream not available.")
                commons.spin(1)
                continue

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""
        frame_time = 1 / self.fps
        t = time.time()
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
                    dur = frame_time - (time.time() - t)
                    if dur > 0:
                        commons.spin(dur)
                else:
                    commons.spin(1)
            except Exception as e:
                print("E:get_frame - ", e)
                if self.capture:
                    self.capture.release()
                self.online = False
                commons.spin(1)
                pass
    
    def update_recognize(self):
        
        if not self.load_video_thread.is_alive():
            self.load_video_thread.start()

        if not self.get_frame_thread.is_alive():
            self.get_frame_thread.start()

        if self.detect_face_thread:
            while(self.detect_face_thread.is_alive()):
                self.detect_face_thread_wait_stop = True
                commons.spin(0.5)
                
        self.detector_backend = settings.get("PROCESSING", "DETECTED_METHOD", fallback="retinaface")
        self.wait_recognize = settings.get("PROCESSING", "WAIT_RECOGNIZED", fallback="True") == "True"
        
        try:
            self.detected_face_stable = int(settings.get("PROCESSING", "DETECTED_FACE_STABLE", fallback="2"))
        except Exception as e:
            logger.warning("Invalid DETECTED_FACE_STABLE, using default 2: %s", e)
            self.detected_face_stable = 2

        self.detected_face_tracking = settings.get("PROCESSING", "DETECTED_FACE_TRACKING", fallback="False") == "True" if self.face_tracking is None else self.face_tracking

        print(f'detector: {self.detector_backend}, slow: {self.wait_recognize}, stable: {self.detected_face_stable}')

        self.detect_face_thread = Thread(target=self.detect_face, args=())
        self.detect_face_thread.daemon = True
        self.detect_face_thread_wait_stop = False
        self.detect_face_thread.start()

    def get_largest_face(self, faces):
        if not faces: return None
        return max(faces, key=lambda face: (face["facial_area"]["w"] * face["facial_area"]["h"]))

    def calculate_face_iou(self, face, face_last):
        x1, y1, w1, h1 = face["x"], face["y"], face["w"], face["h"]
        x2, y2, w2, h2 = face_last["x"], face_last["y"], face_last["w"], face_last["h"]

        # Calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = w1 * h1
        boxBArea = w2 * h2

        # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def detect_face(self):
        """get face from frame""" 
        # build models once to store them in the memory
        # otherwise, they will be built after cam started and this will cause delays 
        num_frames_with_faces = 0
        while True:            
            if self.detect_face_thread_wait_stop:
                break
            if self.wait_recognize and len(self.faces) >0: 
                commons.spin(1)
                continue
            if len(self.deque) < 1:
                commons.spin(2)
                continue
            try:
                t = time.time()
                frame = (self.deque[-1]).copy()
                face_objs = DeepFace.extract_faces(
                    img_path=frame.copy(),
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    # color_face="bgr",
                    align=True
                ) 
                dur = time.time() - t
                # print("Face detection time: ", t)
                face_bigger = self.get_largest_face(face_objs) 
                
                if face_bigger and face_bigger.get("confidence", 0) >= self.face_confidence:
                    item = face_bigger.copy()
                    item['face_crop'] = frame[item['facial_area']['y']: item['facial_area']['y'] + item['facial_area']['h'], item['facial_area']['x']: item['facial_area']['x'] + item['facial_area']['w']] 
                    item['frame'] = frame.copy()
                    item["tic"] = time.time()
                    item["detect_time"] = dur

                    face_last = self.last_face[-1] if len(self.last_face) > 0 else None
                    item["iou"] = self.calculate_face_iou(item['facial_area'], face_last['facial_area']) if face_last is not None else 0
                    
                    if((num_frames_with_faces > 0 and item["iou"] < 0.5) or num_frames_with_faces <1):
                        # if the detected face is very different from the last detected face, reset the counter, this can help reduce false positives when using retinaface detector
                        num_frames_with_faces = 1
                        # logger.warning("Face detected with low iou area %s, reset num_frames_with_faces counter", item.get("iou", 0))
                        item["seq_id"] = str(uuid.uuid4())
                    else:
                        num_frames_with_faces = num_frames_with_faces + 1
                        item["seq_id"] = face_last.get("seq_id", str(uuid.uuid4())) if face_last is not None else str(uuid.uuid4())

                    item["num_frames_with_faces"] = num_frames_with_faces
                    # logger.info("Detect face seq %s: time: %s s", item["num_frames_with_faces"], item['detect_time'])
                    self.last_face.append(item)
                    # only append face if it has been detected in consecutive frames or has high confidence, this can help reduce false positives when using retinaface detector
                    if num_frames_with_faces % self.detected_face_stable == 0 and item.get("iou", 0) >= 0.7:
                        if self.detected_face_tracking:
                            # track face in the stream, return the latest position of the face
                            face_recognized = self.face_recognized[-1] if len(self.face_recognized) > 0 else None
                            face_recognized_seq_id = face_recognized.get("seq_id", "") if face_recognized is not None else ""
                            if face_recognized is None or (face_recognized is not None and face_recognized_seq_id != item["seq_id"]):
                                self.faces.append(item.copy())
                        else:
                            self.faces.append(item.copy())
                else:
                    print("Face detection failed: %s", time.time())
                    num_frames_with_faces = 0
                    self.last_face.append(None)
                dur = time.time() - t
                if dur > 1:
                    logger.warning("Face detection time is too long: %s s", dur)

            except Exception as e:
                print("Detect face e: ", e)
                logger.error("Error in detect_face: %s", str(e))
                commons.spin(1)
                pass 
            finally:
                t = time.time()

    showing_face_seq = ""
    
    def _safe_get(self, obj, *keys, default=None):
        try:
            cur = obj
            for key in keys:
                if cur is not None and isinstance(cur, dict):
                    cur = cur.get(key, None)
                elif hasattr(cur, key):
                    cur = getattr(cur, key)
                else:
                    return default
            return cur
        except Exception as e:
            print("Error in _safe_get: ", e)
            return default

    def get_face_note_text(self,face):
        face_recognized_item = self.face_recognized[-1] if len(self.face_recognized) > 0 else None
        face_recognized_seq_id = face_recognized_item.get("seq_id", "") if face_recognized_item is not None else ""
        face_recognized = self._safe_get(face_recognized_item, "recognized")
        recognized_msv = self._safe_get(face_recognized_item, "recognized", "payload", "msv")

        face_seq_id = face.get("seq_id", "") if face is not None else ""
        seq_last8 = face_seq_id[-8:] if face_seq_id else ""
        
        if face_recognized is not None and face_recognized_seq_id == face_seq_id:
            return "Recognized: {}".format(recognized_msv or seq_last8)
                
        if face is not None: 
            return "Face {}: {}".format(seq_last8, face["num_frames_with_faces"])
        return ""

    def set_frame(self):
        """Sets pixmap image to video frame"""
        if not self.online:
            def create_connecting_image(width, height):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                
                text = "Find Camera Source{}".format("." * (self.load_video_thread_count % 4))
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
            commons.spin(1)
            return

        face = self.last_face[-1] if len(self.last_face) > 0 else None
        seq_id = face.get("seq_id", "") if face is not None else ""
        face_area = face['facial_area'] if face is not None else None
        frame = None
        if self.deque and self.online:
            # Grab latest frame
            frame = (self.deque[-1]).copy()
            
        if self.wait_recognize and face is not None:
            # show "wait recognize" text if face is detected but not yet recognized
            frame = face.get("frame", None)
            
        if frame is not None:
            if face is not None and (time.time() - face['tic'] < 2000 or self.wait_recognize): # only draw face box if detected in the last 2 seconds
                x, y, w, h = face_area["x"], face_area["y"], face_area["w"], face_area["h"]
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (67, 67, 67), 2
                )  # draw rectangle to main image
                #draw text under face box
                cv2.rectangle(frame, (x, y + h), (x + w, y + h + 34), color=(67,67,67), thickness=-1)
                cv2.putText(frame, self.get_face_note_text(face), (x+10, y + h + 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)
            # Keep frame aspect ratio
            if self.maintain_aspect_ratio:
                frame = imutils.resize(frame, width=self.screen_width)
            # Force resize
            else:
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))

            # Add timestamp to cameras
            cv2.rectangle(frame, (self.screen_width-190,0), (self.screen_width,50), color=(0,0,0), thickness=-1)
            cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (self.screen_width-185,37), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), lineType=cv2.LINE_AA)

            # Convert to pixmap and set to video frame
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)

        if face is not None and self.detected_frame is not None: 
            if not hasattr(self, "detected_frame_tic"):
                self.detected_frame_tic = 0
            if self.detected_frame_tic != face["tic"]:
                self.detected_frame_tic = face["tic"]
                x, y, w, h = face_area["x"], face_area["y"], face_area["w"], face_area["h"]
                img_face = face["frame"][y: y + h, x: x + w]  # crop detected face
                img_face = imutils.resize(img_face, width=200)
                # cv2.rectangle(img_face, (0, 0), (100,20), color=(66, 66, 66), thickness=-1)
                cv2.putText(img_face, "Face in stream", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)
                img_face = QtGui.QImage(img_face, img_face.shape[1], img_face.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                pix_face = QtGui.QPixmap.fromImage(img_face)
                self.detected_frame.setPixmap(pix_face)

        if face is not None and seq_id != self.showing_face_seq:
            self.showing_face_seq = seq_id
    
    def get_video_frame(self):
        return self.video_frame
    
    def get_face_detected_frame(self):        
        if self.detected_frame is None:
            self.detected_frame = QtWidgets.QLabel()
        return self.detected_frame