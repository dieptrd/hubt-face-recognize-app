from PyQt5.QtWidgets import *
from appSettings import settings

class ImportDialog(QDialog):
    def __init__(self, parent=None):
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
        classList = settings.class_name()

        layout = QFormLayout()
        self.className = QLineEdit(classList[0] if len(classList) > 0 else "", self)
        layout.addRow('Class Name:', self.className) 

        self.studentId = QLineEdit("", self)  
        layout.addRow('Student ID:', self.studentId)
        
        self.studentName = QLineEdit("", self)
        layout.addRow('Student Name:', self.studentName)
        widget.setLayout(layout)
        return widget
