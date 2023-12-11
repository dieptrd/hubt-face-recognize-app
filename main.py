from PyQt5.QtWidgets import *
import sys 
from collections import deque
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
        self.setWindowTitle("Widgets App")

        layout = QVBoxLayout() 

        # Stream links
        camera0 = 0
        
        # Create camera widgets
        print('Creating Camera Widgets...')
        zero = CameraWidget(700,800, faces, camera0 , aspect_ratio=True)
        # Add widgets to layout
        print('Adding Camera widget to layout...')
        layout.addWidget(zero.get_video_frame()) 
        print('Verifying camera credentials...') 
        recognize = FaceRecognize(faces)
        print('Adding Faces recognize widget to layout...')
        layout.addWidget(recognize.get_view())
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()