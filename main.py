from PyQt5.QtWidgets import *
import sys 
from collections import deque
from appSettings import settings
from settingDialog import SettingDialog
from cameraWidget import CameraWidget
from FaceRecognize import FaceRecognize
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
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
        # Stream links
        camera0 = 0
        
        # Create camera widgets
        print('Creating Camera Widgets...')
        zero = CameraWidget(500,600, faces, camera0 , aspect_ratio=True)
        recognize = FaceRecognize(faces)
        self.recognize = recognize
        self.camera = zero
        # Add widgets to layout
        print('Adding Camera widget to layout...')
        layout.addWidget(zero.get_video_frame())  
        print('Verifying camera credentials...') 
        
        print('Adding Faces recognize widget to layout...')
        layout.addWidget(self.init_regcognize_frame())
        layout.addWidget(recognize.get_view())
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
    
    def onSettingClick(self, s):
        dlg = SettingDialog(self)
        result = dlg.exec()
        if result:
            dlg.updateChanged()
            if hasattr(self, 'recognize'):
                self.recognize.load_faces()
                pass
            if hasattr(self, 'camera'):
                self.camera.update_recognize()
                pass 
        print("dialog result: ", result)

    def init_regcognize_frame(self):
        _widget = QWidget(self)
        _widget.setLayout(QHBoxLayout())
        _widget.layout().addWidget(self.camera.get_face_detected_frame())
        _widget.layout().addWidget(self.recognize.get_recognize_frame())

        return _widget

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()