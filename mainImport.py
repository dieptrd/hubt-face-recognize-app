import os
import sys
from collections import deque
from threading import Thread


from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QToolBar, QAction, QWidget, QHBoxLayout, QLabel

from deepface import DeepFace

from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings
from settingDialog import SettingDialog
from selectClass import SelectClass
from addNewStudent import AddNewStudent
from cameraWidget import CameraWidget
from faceRecognize import FaceRecognize
from logger import logger
from dbProvider import db
import commons
import logging
from PyQt5 import QtCore

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"  

class MainWindow(QMainWindow):

    """
    The main window of the application.

    Attributes:
        camera (CameraWidget): The camera widget for capturing video.
        recognize (FaceRecognize): The face recognition widget.
    """

    def __init__(self):
        super().__init__()
        global faces, faces_recognized, faces_new
        faces = deque(maxlen=1)
        faces_recognized = deque(maxlen=1)
        faces_new = deque(maxlen=1)
        self.setWindowTitle("Face from Camera") 
        layout = QVBoxLayout() 
        #Toolbar
        toolbar = QToolBar("Dieptrd")
        self.addToolBar(toolbar)
        button_action = QAction("Setting", self)
        button_action.setStatusTip("Change app's setting")
        button_action.triggered.connect(self.onSettingClick)
        toolbar.addAction(button_action) 
        # open settingDialog when first time start app
        first_run =settings.get("GLOBAL", "FIRST_RUN", fallback="0")
        if first_run == "0":
            settings.set("GLOBAL", "FIRST_RUN", "1")
            settings.update()
            self.onSettingClick()

        # Create "Add New Student" button and place it above the recognize control
        self.new_student_button = QtWidgets.QPushButton("Add New Student", self)
        self.new_student_button.setToolTip("Add a new student to the database") 
        self.new_student_button.clicked.connect(self._on_new_student)
        
        self.clear_current_faces_button = QtWidgets.QPushButton("Clear Current Faces", self)
        self.clear_current_faces_button.setToolTip("Clear current detected faces in the camera")
        self.clear_current_faces_button.clicked.connect(self._on_clear_current_faces)

        # Create camera widgets
        logger.debug('Creating Camera Widgets...')

        self.camera = CameraWidget(520,600, faces, faces_recognized, face_tracking=False, aspect_ratio=True)
        self.recognize = FaceRecognize(
            faces, 
            faces_recognized, 
            face_new=faces_new, 
            face_show_type="new",
            recognize_score_threshold=0.8
        )

        self.faces_new_thread = Thread(target=self.recognize_new_face_detection, args=(), daemon = True)
        self._wait_stop = False
        self.faces_new_thread.start()
        # Add widgets to layout
        logger.debug('Adding Camera and Faces recognize widget to layout...')
        layout.addWidget(self.init_regcognize_video_frame())  
        logger.debug('Verifying camera credentials...')  

        layout.addWidget(self.recognize.get_new_faces_view())
        
        #show progress dialog
        # self.loading_thread()
        
        #add logging textbox
        self.text_log = QtWidgets.QTextEdit(self)
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log)
        
        class QTextEditLogger(QtCore.QObject, logging.Handler):
            text_signal = QtCore.pyqtSignal(str)
            def __init__(self, text_edit):
                QtCore.QObject.__init__(self)
                logging.Handler.__init__(self)
                self.text_edit = text_edit
                self.text_signal.connect(self.append_text)

            def emit(self, record):
                try:
                    msg = self.format(record)
                except Exception:
                    msg = str(record)
                self.text_signal.emit(msg)

            def append_text(self, msg):
                self.text_edit.append(msg)
                sb = self.text_edit.verticalScrollBar()
                sb.setValue(sb.maximum())

        qt_handler = QTextEditLogger(self.text_log)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(qt_handler) 
        logger.setLevel(logging.WARNING)

        logger.info("Logging to text_log initialized.")
        
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        
    def showEvent(self, event):
        super().showEvent(event)
        #show progress dialog
        #Face data clear
        db.reload(True)
        
        if hasattr(self, 'camera'):
            self.camera.update_recognize()
        if hasattr(self, 'recognize'):
            self.recognize.reload_recognize_thread()
        
    
    def closeEvent(self, event):
        """Called when the main window is closing."""
        self._wait_stop = True
        self.faces_new_thread.join()
        try:
            if hasattr(self, 'camera'):
                # Ensure CameraWidget threads stop even if CameraWidget.closeEvent
                # is not triggered during app shutdown.
                self.camera.stop()
                # self.camera.close()
        finally:
            super().closeEvent(event)

        event.accept()
        
    def _on_clear_current_faces(self):
        db.get_client().clear().reload()
        if hasattr(self, 'recognize'):
            self.recognize.clear_new_faces_view()
        logger.info("Cleared current faces in client and updated recognize widget.")

    def _on_new_student(self):
        def calculate_frontal_score(face_data):
            """
            Tính toán điểm số nhìn thẳng dựa trên độ đối xứng của mắt.
            Score càng gần 0, mặt càng thẳng tuyệt đối.
            """
            face_area = commons._safe_get(face_data,"payload","face_area", default={})
            
            # 1. Lấy tọa độ mắt (x, y)
            # Lưu ý: JSON của bạn định dạng [0: x_val, 1: y_val]
            left_eye = face_area.get("left_eye", [])
            right_eye = face_area.get("right_eye", [])
            
            if not left_eye or not right_eye:
                return float('inf') # Không đủ dữ liệu điểm mốc
                
            lx, ly = left_eye[0], left_eye[1]
            rx, ry = right_eye[0], right_eye[1]
            
            # 2. Xác định tâm trục dọc của khuôn mặt (Face Center X)
            # Dựa vào bounding box: x + w/2
            face_center_x = face_area["x"] + (face_area["w"] / 2)
            
            # 3. Tính khoảng cách ngang từ tâm mặt tới mỗi mắt
            dist_to_left = abs(face_center_x - lx)
            dist_to_right = abs(face_center_x - rx)
            
            # 4. Tính toán độ lệch đối xứng (Yaw Score)
            # Nếu mặt thẳng tuyệt đối: dist_to_left == dist_to_right -> yaw_score = 0
            total_dist = dist_to_left + dist_to_right
            if total_dist == 0:
                return 0
                
            yaw_score = abs(dist_to_left - dist_to_right) / total_dist
            return yaw_score

        dlg = AddNewStudent(self)
        result = dlg.exec()
        print("Add new student dialog result: %s", result)
        if result:
            # reload faces database and refresh UI
            try:
                info = dlg.get_info()
                if not info:
                    logger.debug("No new student info returned from dialog")
                    return

                print("New student info: ", info)
                
                (faces, _) = db.get_client().load_all_faces() or ([],0)
                if not faces:
                    logger.debug("No faces in client to update")
                    return
 
                if len(faces) > 0:
                    print("Uploading {} face(s) to DB".format(len(faces)))
                    upload_vectors = []
                    
                    face = min(faces, key= lambda f: calculate_frontal_score(f))
                    print(face)
                    id = commons._safe_get(face,"id", default=None)
                    payload = {
                        "msv": commons._safe_get(info, "msv", default=""),
                        "fullname": commons._safe_get(info, "fullname", default=""),
                        "tel": commons._safe_get(info, "tel", default=""),
                        "face": commons._safe_get(face, "payload", "face", default=None),
                        "face_area": commons._safe_get(face, "payload", "face_area", default=None),
                        "frame": commons._safe_get(face, "payload", "frame", default=None)
                    }
                    for face in faces:
                        upload_vectors.extend(face.vector)
                        
                    print("Prepared {} face(s) for upload".format(len(upload_vectors)))
                    db.get_db().upsert_face(id, upload_vectors, payload)
                    db.get_client().clear().reload()
                    if hasattr(self, 'recognize'):
                        self.recognize.clear_new_faces_view()
                else:
                    print("No faces to upload after processing")
            except Exception as e:
                print("Failed reloading DB after adding new student: {}".format(e)) 
    
    def onSettingClick(self):
        dlg = SettingDialog(self)
        result = dlg.exec()
        if result:
            dlg.updateChanged()
            self.loading_thread()
        logger.debug(f"dialog result: {result}")

    def init_regcognize_video_frame(self):
        """
        Initialize the layout for the camera video frame and the face detection frame.

        Returns:
            QWidget: The widget containing the camera video frame and the face detection frame.
        """
        _widget = QWidget(self)
        _widget.setLayout(QHBoxLayout())
        _widget.layout().addWidget(self.camera.get_video_frame())
        _widget.layout().addWidget(self.init_regcognize_frame()) 
        return _widget

    def init_regcognize_frame(self):
        """
        Initialize the layout for the face detection label and the recognized face frame.

        Returns:
            QWidget: The widget containing the face detection label and the recognized face frame.
        """
        _widget = QWidget(self)
        _widget.setLayout(QVBoxLayout()) 
        _widget.layout().addWidget(self.clear_current_faces_button)
        _widget.layout().addWidget(self.new_student_button)
        # _widget.layout().addWidget(self.camera.get_face_detected_frame())
        _widget.layout().addWidget(self.recognize.get_recognize_frame())        
        return _widget

    def loading_thread(self):
        """
        Simulate a loading process by updating the progress dialog.

        This method is intended to be run in a separate thread to avoid blocking the main UI thread.
        It updates the progress dialog with a simulated loading process.
        """
        
        self.progress_dialog = QtWidgets.QProgressDialog()
        self.progress_dialog.setRange(0, 1000)
        self.progress_dialog.setModal(True)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.show() 

        # Face model loading process
        self.progress_dialog.setLabelText("Face Model Loading...")
        self.progress_dialog.setValue(0)  # Update progress to 10%
        self.model_name = settings.get("PROCESSING", "recognize_method", fallback="VGG-Face")
        DeepFace.build_model(model_name=self.model_name)

        self.progress_dialog.setValue(300)  # Update progress to 30%
        #Face data loading process
        self.progress_dialog.setLabelText("Face Data Loading...")
        db.reload_db(True)
        
        if hasattr(self, 'camera'):
            self.camera.update_recognize()
        if hasattr(self, 'recognize'):
            self.recognize.reload_recognize_thread()

        self.progress_dialog.close()

    def recognize_new_face_detection(self):
        """
        Continuously check for new faces detected and update the recognize widget.

        This method runs in a separate thread and checks for new faces detected by the camera.
        If a new face is detected, it updates the recognize widget with the new face information.
        """
        _id = ""
        while not self._wait_stop: 
            if len(faces_new) > 0:
                print("New face detected.")
                (id, vector, payload) = faces_new.pop()
                db.get_client().upsert_face(id, vector, payload)
            else:
                commons.spin(0.2)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()