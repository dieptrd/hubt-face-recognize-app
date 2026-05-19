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
        # set the window width (adjust as needed)
        self.setFixedWidth(520)
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
        self.msv.setPlaceholderText("MSV (auto-uppercase)")
        def _on_msv_changed(text):
            new = text.upper()
            if new != text:
                pos = self.msv.cursorPosition()
                self.msv.blockSignals(True)
                self.msv.setText(new)
                self.msv.blockSignals(False)
                self.msv.setCursorPosition(min(pos, len(new)))
        self.msv.textChanged.connect(_on_msv_changed)
        
        self.fullname = QLineEdit("", self)
        self.fullname.setPlaceholderText("Full name (Each word starts with uppercase)")

        def _on_fullname_changed(text):
            # Preserve spacing while capitalizing only the first character of each word
            parts = text.split(' ')
            new_parts = []
            for p in parts:
                if p:
                    new_parts.append(p[0].upper() + p[1:])
                else:
                    new_parts.append(p)
            new = ' '.join(new_parts)
            if new != text:
                pos = self.fullname.cursorPosition()
                self.fullname.blockSignals(True)
                self.fullname.setText(new)
                self.fullname.blockSignals(False)
                self.fullname.setCursorPosition(min(pos, len(new)))

        self.fullname.textChanged.connect(_on_fullname_changed)
        
        self.tel = QLineEdit("", self)
        # restrict tel to 10 digits and provide live validation + block dialog accept if invalid
        validator = QtGui.QRegExpValidator(QtCore.QRegExp(r'\d{10}'))
        self.tel.setValidator(validator)
        self.tel.setMaxLength(10)
        self.tel.setPlaceholderText('10 digits phone number')

        def _update_tel_style(text):
            if len(text) == 10 and text.isdigit():
                self.tel.setStyleSheet("")  # valid
            else:
                self.tel.setStyleSheet("border: 1px solid red;")

        self.tel.textChanged.connect(_update_tel_style)
        _update_tel_style(self.tel.text())

        # override accept to validate before closing
        _original_accept = self.accept

        def _accept_override():
            text = self.tel.text().strip()
            if not (len(text) == 10 and text.isdigit()):
                QMessageBox.warning(self, "Invalid phone", "Tel must be a 10-digit number.")
                return
            _original_accept()

        self.accept = _accept_override
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
            "msv": self.msv.text().strip().upper(),
            "fullname": self.fullname.text(),
            "tel": self.tel.text(),
        }
