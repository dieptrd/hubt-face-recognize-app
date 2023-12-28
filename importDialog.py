from PyQt5.QtWidgets import *
from appSettings import settings

class ImportDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)   

        self.setWindowTitle("Add new Student") 

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self._init_DBWidget()) 
        layout.addWidget(self.buttonBox) 

        self.setLayout(layout)  

    def _init_DBWidget(self):
        widget = QGroupBox("Add payload info.", self)

        layout = QFormLayout()
        self.studentId = QLineEdit("", self)  
        layout.addRow('Student ID:', self.studentId)

        widget.setLayout(layout)
        return widget
