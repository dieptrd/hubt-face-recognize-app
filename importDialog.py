from PyQt5.QtWidgets import *
from appSettings import settings

class ImportDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)   

        self.setWindowTitle("Setting") 

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self._init_DBWidget()) 
        layout.addWidget(self.buttonBox) 

        self.setLayout(layout)

    def _init_RecognizeWidget(self):
        widget = QGroupBox("Face Recognize Methods", self)
        layout = QFormLayout()

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
        layout.addRow('Face Detect Methods', self.recognize)

        widget.setLayout(layout)
        return widget
    def detectChanged(self, e):
        print(e)
        self.detect_method = self.detected.currentText()
    def recognizeChanged(self, e):
        print(e)
        self.recognize_method = self.detected.currentText()

    def _init_DBWidget(self):
        widget = QGroupBox("Vector DB Location", self)

        layout = QFormLayout()
        self.studentId = QLineEdit("", self)  
        layout.addRow('Student ID:', self.studentId)

        widget.setLayout(layout)
        return widget
    def updateChanged(self):
        settings.set("VECTORDB", "HOST", self.host.text())
        settings.set("VECTORDB", "PORT", self.port.text())
        settings.set("PROCESSING", "WAIT_RECOGNIZED", self.process.currentText())
        settings.set("PROCESSING", "DETECTED_METHOD", self.detect_method)
        settings.set("PROCESSING", "RECOGNIZE_METHOD", self.recognize_method)
        settings.update()
