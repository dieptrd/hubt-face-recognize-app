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
from FaceRecognize import FaceRecognize
from logger import logger
from db import db
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
        
        #show progress dialog
        self.loading_thread()

        # Create camera widgets
        logger.debug('Creating Camera Widgets...')

        self.camera = CameraWidget(520,600, faces, faces_recognized, face_tracking=False, aspect_ratio=True)
        self.recognize = FaceRecognize(faces, faces_recognized, face_new=faces_new)

        self.faces_new_thread = Thread(target=self.recognize_new_face_detection, args=())
        self.faces_new_thread.daemon = True
        self.faces_new_thread_wait_stop = False
        self.faces_new_thread.start()
        # Add widgets to layout
        logger.debug('Adding Camera and Faces recognize widget to layout...')
        layout.addWidget(self.init_regcognize_video_frame())  
        logger.debug('Verifying camera credentials...')  

        layout.addWidget(self.recognize.get_new_faces_view())
        
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
        qt_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(qt_handler)
        if logger.level == 0:
            logger.setLevel(logging.INFO)

        logger.info("Logging to text_log initialized.")
        
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

    def _on_new_student(self):
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

                print("New student info: msv={} fullname={}".format(getattr(info, "msv", None), getattr(info, "fullname", None)))
                
                faces = db.get_all_faces_client() or []
                if not faces:
                    logger.debug("No faces in client to update")
                    return
 
                if len(faces) > 0:
                    print("Uploading {} face(s) to DB".format(len(faces)))
                    upload_faces = []
                    for face in faces:
                        payload = {
                            "msv": commons._safe_get(info, "msv", default=""),
                            "fullname": commons._safe_get(info, "fullname", default=""),
                            "tel": commons._safe_get(info, "tel", default=""),
                            "face": commons._safe_get(face, "payload", "face", default=None),
                            "face_area": commons._safe_get(face, "payload", "face_area", default=None),
                            "frame": commons._safe_get(face, "payload", "frame", default=None)
                        }
                        upload_faces.append(PointStruct(
                            id=face.id,
                            vector=face.vector,
                            payload=payload
                        ))
                    print("Prepared {} face(s) for upload".format(len(upload_faces)))
                    db.upsert_face_db(upload_faces)
                    db.clear_client()
                else:
                    print("No faces to upload after processing")
            except Exception as e:
                print("Failed reloading DB after adding new student: {}".format(e))
                
            if hasattr(self, 'camera'):
                self.camera.update_recognize()
            if hasattr(self, 'recognize'):
                self.recognize.reload_recognize_thread()
    
    def onSettingClick(self):
        dlg = SettingDialog(self)
        result = dlg.exec()
        if result:
            dlg.updateChanged()
            self.loading_thread()
        logger.debug("dialog result: %s", result)

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
        self.db = db.reload_db()
        self.db.load_all_faces_to_client_with_filter()
        
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
        while True:
            if self.faces_new_thread_wait_stop:
                logger.debug("Stopping recognize_new_face_detection thread.")
                break
            if len(faces_new) > 0:
                print("New face detected.")
                (id, vector, payload) = faces_new.pop()
                db.upsert_face_client(id, vector, payload)
            else:
                QtCore.QThread.msleep(100)  # Sleep briefly to avoid busy waiting

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()