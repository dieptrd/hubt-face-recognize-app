from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets, QtGui

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings
import time
import threading
from db import db
import commons

class AddNewStudent(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)   

        self.setWindowTitle("Add New Student...")

        self.collection_name= settings.get("VECTORDB","COLLECTION_NAME", fallback="hubt_faces")

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self._db = None
        self.delay_timer = None
        
        layout = QVBoxLayout(self)
        layout.addWidget(self._init_DBWidget()) 

        layout.addWidget(self._init_facesWidget())

        layout.addWidget(self.buttonBox)

        self.logs = QLabel("", self)
        layout.addWidget(self.logs)
        self.setLayout(layout)  

    def _init_DBWidget(self):
        widget = QGroupBox("Base Student Information", self)
        layout = QFormLayout()
        self.msv = QLineEdit("", self)
        self.fullname = QLineEdit("", self)
        self.tel = QLineEdit("", self)
        layout.addRow('MSV:', self.msv)
        layout.addRow('Fullname:', self.fullname)
        layout.addRow('Tel:', self.tel)
        widget.setLayout(layout)
        return widget
    
    def _init_facesWidget(self):
        widget = QGroupBox("Student's Faces", self)
        layout = QVBoxLayout()
        self.facesWidget = QtWidgets.QTextEdit()
        self.facesWidget.setReadOnly(True)
        layout.addWidget(self.facesWidget)
        widget.setLayout(layout)
        faces = db.get_all_faces_client()
        for faceitem in faces:
            face = commons._safe_get(faceitem, "payload", "face", default=None)
            self.facesWidget.insertHtml("<img src='{}'> {}".format(face, " ")) 
        self.facesWidget.append("Total faces: {}".format(len(faces)))
        return widget

    def get_info(self):
        return {
            "msv": self.msv.text(),
            "fullname": self.fullname.text(),
            "tel": self.tel.text(),
        }
