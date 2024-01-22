import os
import sys
from collections import deque

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QToolBar, QAction, QWidget, QHBoxLayout, QLabel

from appSettings import settings
from settingDialog import SettingDialog
from cameraWidget import CameraWidget
from FaceRecognize import FaceRecognize
from logger import logger

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
        global faces 
        faces = deque(maxlen=1)
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
            self.onSettingClick(None)
        
        # Create camera widgets
        logger.debug('Creating Camera Widgets...')
        
        self.camera = CameraWidget(500,600, faces, aspect_ratio=True)
        self.recognize = FaceRecognize(faces)
        
        # Add widgets to layout
        logger.debug('Adding Camera and Faces recognize widget to layout...')
        layout.addWidget(self.init_regcognize_video_frame())  
        logger.debug('Verifying camera credentials...')  

        layout.addWidget(self.recognize.get_view())
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
    
    def onSettingClick(self):
        dlg = SettingDialog(self)
        result = dlg.exec()
        if result:
            dlg.updateChanged()
            if hasattr(self, 'recognize'):
                self.recognize.load_faces()
            if hasattr(self, 'camera'):
                self.camera.update_recognize()
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
        _widget.layout().addWidget(self.camera.get_face_detected_frame())
        _widget.layout().addWidget(self.recognize.get_recognize_frame())        
        return _widget

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()