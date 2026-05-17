import os
import sys
from collections import deque

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QToolBar, QAction, QWidget, QHBoxLayout, QLabel

from deepface import DeepFace

from appSettings import settings
from settingDialog import SettingDialog
from selectClass import SelectClass
from cameraWidget import CameraWidget
from faceRecognize import FaceRecognize
from logger import logger
import logging
from db import db

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"  

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


class LevelFilter(logging.Filter):
    """Allow only WARNING and ERROR level records."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in (logging.WARNING, logging.ERROR)

class MainWindow(QMainWindow):
    """
    The main window of the application.

    Attributes:
        camera (CameraWidget): The camera widget for capturing video.
        recognize (FaceRecognize): The face recognition widget.
    """

    def __init__(self):
        super().__init__()
        global faces, faces_recognized
        faces = deque(maxlen=1)
        faces_recognized = deque(maxlen=1)
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
            
        # show select class dialog
        # dialog = SelectClass(self)
        # dialog.exec()

        # Create camera widgets
        logger.debug('Creating Camera Widgets...')

        self.camera = CameraWidget(520,600, faces, faces_recognized, aspect_ratio=True)
        self.recognize = FaceRecognize(faces, faces_recognized) 
        
        #show progress dialog
        self.loading_thread()
        
        # Add widgets to layout
        logger.debug('Adding Camera and Faces recognize widget to layout...')
        layout.addWidget(self.init_regcognize_video_frame())

        # layout.addWidget(self.recognize.get_new_faces_view())
        #add logging textbox
        self.text_log = QtWidgets.QTextEdit(self)
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log) 

        qt_handler = QTextEditLogger(self.text_log)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        # Only show INFO and ERROR levels in the UI text box
        qt_handler.addFilter(LevelFilter())
        logger.addHandler(qt_handler)
        # Keep a reference to the handler so we can remove it on close
        self._qt_handler = qt_handler
        # Ensure logger will emit INFO records
        logger.setLevel(logging.INFO)

        logger.info("Logging to text_log initialized.")
        
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
    
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
        # _widget.layout().addWidget(self.camera.get_face_detected_frame())
        _widget.layout().addWidget(self.recognize.get_recognize_frame(show_info=True))        
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
        self.db = db.reload_db(True)
        self.db.load_all_faces_to_client_with_filter()
        
        if hasattr(self, 'camera'):
            self.camera.update_recognize()
        if hasattr(self, 'recognize'):
            self.recognize.reload_recognize_thread()

        self.progress_dialog.close()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()