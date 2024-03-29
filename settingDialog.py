import os
from PyQt5.QtWidgets import *
from appSettings import settings

class SettingDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent) 

        self.detect_method = settings.get("PROCESSING", "DETECTED_METHOD", fallback="retinaface")
        self.recognize_method = settings.get("PROCESSING", "RECOGNIZE_METHOD", fallback="VGG-Face")

        self.setWindowTitle("Setting") 

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self._init_DBWidget())
        layout.addWidget(self._init_RecognizeWidget())
        layout.addWidget(self.buttonBox) 

        self.setLayout(layout)

    def _init_RecognizeWidget(self):
        widget = QGroupBox("Face Recognize Methods", self)
        layout = QFormLayout()

        
        layout.addRow('Deepface Home:', self._init_DeepfaceWidget())

        wait = settings.get("PROCESSING", "WAIT_RECOGNIZED", fallback="True")
        self.process = QComboBox(self)
        self.process.addItems(["True", "False"])
        self.process.setCurrentText(wait)
        layout.addRow('Wait recognized after detect face:', self.process)  

        self.detected = QComboBox(self)
        self.detected.addItems("opencv, retinaface, mtcnn, ssd, dlib, mediapipe, yolov8".split(", "))
        self.detected.setCurrentText(self.detect_method)
        self.detected.currentIndexChanged.connect(self.detectChanged)
        layout.addRow('Face Detect Methods', self.detected)

        self.recognize = QComboBox(self)
        self.recognize.addItems("VGG-Face".split(", ")) # VGG-Face, Facenet, OpenFace, DeepFace, DeepID
        self.recognize.setCurrentText(self.recognize_method)
        self.recognize.currentIndexChanged.connect(self.recognizeChanged)
        layout.addRow('Face Recognize Methods', self.recognize)

        time_windows = str(10) # 10 seconds
        self.time_windows = QLineEdit(time_windows,self)
        layout.addRow('Recognize In Stream Time Windows:', self.time_windows)

        widget.setLayout(layout)
        return widget
    def detectChanged(self, e):
        print(e)
        self.detect_method = self.detected.currentText()
    def recognizeChanged(self, e):
        print(e)
        self.recognize_method = self.detected.currentText()
    def changeDeepfaceHome(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.home.setText(folder_path)
    def _init_DBWidget(self):
        widget = QGroupBox("Vector DB Location", self)

        layout = QFormLayout()
        self.host = QLineEdit(settings.get("VECTORDB", "HOST", fallback="localhost"), self)
        self.port = QLineEdit(str(settings.get("VECTORDB", "PORT", fallback=6333)), self) 
        layout.addRow('Vector Database Host:', self.host)
        layout.addRow('Vector Database Port:', self.port)
        widget.setLayout(layout)
        return widget
    def _init_DeepfaceWidget(self):
        widget = QWidget(self)
        layout = QHBoxLayout()
        self.home = QLineEdit(settings.get_deepface_home(), self)
        self.home.setReadOnly(True)
        layout.addWidget(self.home)

        btn = QPushButton("Change", self)
        btn.clicked.connect(self.changeDeepfaceHome)        
        layout.addWidget(btn)

        widget.setLayout(layout)
        return widget
    def updateChanged(self):
        settings.set("VECTORDB", "HOST", self.host.text())
        settings.set("VECTORDB", "PORT", self.port.text())
        settings.set("PROCESSING", "WAIT_RECOGNIZED", self.process.currentText())
        settings.set("PROCESSING", "DETECTED_METHOD", self.detect_method)
        settings.set("PROCESSING", "RECOGNIZE_METHOD", self.recognize_method)
        settings.set("PROCESSING", "TIME_WINDOWS", self.time_windows.text())
        #update deepface home folder
        folder_path = self.home.text()
        # check if folder_path is exists
        if os.path.isdir(folder_path):
            settings.set("GLOBAL", "DEEPFACE_HOME", self.home.text())
        settings.set_deepface_home()
        settings.update()
