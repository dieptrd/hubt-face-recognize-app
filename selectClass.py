from PyQt5.QtWidgets import *

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from appSettings import settings
import time
import threading

class SelectClass(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)   

        self.setWindowTitle("Select Class")
        
        self.collection_name="hubt_faces"

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self._db = None
        self.delay_timer = None
        
        layout = QVBoxLayout(self)
        layout.addWidget(self._init_DBWidget()) 

        self.logs = QLabel("Total records: 0")
        layout.addWidget(self.logs)

        layout.addWidget(self.buttonBox)
        self.setLayout(layout)  

    def _init_DBWidget(self):
        widget = QGroupBox("Select your class name", self)
        classList = settings.class_name()
        layout = QFormLayout()
        self.className = QLineEdit(", ".join(classList), self)
        self.className.textChanged.connect(self.classNameChanged)
        layout.addRow('Class Name:', self.className)
        widget.setLayout(layout)
        return widget
    
    def classNameChanged(self, e):
        
        self.logs.setText("") 

        def delay_and_clear():
            if self.delay_timer:
                # Clear existing delay
                self.delay_timer.cancel()
            # Create new delay
            self.delay_timer = threading.Timer(1, run_code_block)
            self.delay_timer.start()

        def run_code_block():
            # Your code block here            
            text = self.className.text()            
            return self.get_total_recodes(text)

        # Call delay_and_clear() to start the delay
        delay_and_clear()

    
    def get_total_recodes(self, text):
        matchs = [item.strip() for item in text.split(",")]
        matchs.append("undefined")
        
        db = self.get_connection()
        if not db:
            self.logs.setText("Can not connect to DB")
            return 0
        
        offset = 0
        points_count = 0
        while offset != None:
            points, offset = db.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="class",
                            match=models.MatchAny(any=matchs),
                        ),
                    ]
                ),
                offset=offset,
                limit=100,
                with_payload=False,
                with_vectors=False,
            )
            points_count += len(points)
        settings.class_name(text)
        self.logs.setText("Total records: {}".format(points_count))

    def get_connection(self):
        if not self._db:            
            host = settings.get("VECTORDB","HOST", fallback= "localhost")
            port = settings.getint("VECTORDB","PORT", fallback= 6333)
            self._db = QdrantClient(host, port=port)
            try:
                r = self._db.get_collection(self.collection_name)
                # self.logs.setText("Collection status is: {}".format(r.status))
            except:
                self.logs.setText("Can not connect to DB")
                return None
        return self._db
        
