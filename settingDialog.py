from PyQt5.QtWidgets import (
    QFormLayout,
    QDialog, 
    QDialogButtonBox,
    QLineEdit,
    QVBoxLayout,
    QLabel
)
from appSettings import settings

class SettingDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("Setting")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout = QFormLayout()
        self.host = QLineEdit(settings.get("VECTORDB", "HOST", fallback="localhost"), self)
        self.port = QLineEdit(settings.get("VECTORDB", "PORT", fallback=6333), self)
        wait = settings.get("PROCESSING", "WAIT_RECOGNIZED", fallback=True)
        wait_text = "1"
        if not wait:
            wait_text = "0"
        self.process = QLineEdit(wait_text, self)
        layout.addRow('Vector Database Host:', self.host)
        layout.addRow('Vector Database Port:', self.port)
        layout.addRow('Wait recognized after detect face:', self.process) 
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
    def updateChanged(self):
        settings.set("VECTORDB", "HOST", self.host.text())
        settings.set("VECTORDB", "PORT", self.port.text())
        settings.set("PROCESSING", "WAIT_RECOGNIZED", self.process.text()=="1")