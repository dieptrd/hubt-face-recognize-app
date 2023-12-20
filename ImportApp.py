from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import sys 
from collections import deque
from settingDialog import SettingDialog 
import os 
from appSettings import settings
from threading import Thread
from tqdm import tqdm

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from deepface import DeepFace
from deepface.commons import functions, distance

model_name = "VGG-Face"
detector_backend = "opencv"
vector_size = 2622
collection_name="hubt_faces"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
 
# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._thread = None 
        self.percent = deque([0], maxlen=1)
        self.logs = deque(maxlen=10)
        self.setWindowTitle("Add Images to database") 
        layout = QFormLayout()
        _openFolder = QPushButton(self)
        _openFolder.setText("Select")
        _openFolder.clicked.connect(self.selectFolder)

        #Toolbar 
        self.folder_data = ""
        self.folder = QLineEdit("", self)
        self.folder.setReadOnly(True) 
 
        layout.addRow(QLabel('Faces Image Folder:'))
        layout.addRow(self.folder,_openFolder)
        layout.addRow(QLabel("")) 
        
        self.pbar = QProgressBar(self)
        layout.addRow(self.pbar)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_pbar)
        self.timer.start(1)
        
        self.btnScan = QPushButton(self)
        self.btnScan.setText("Import (0 Images)")
        self.btnScan.clicked.connect(self.scanFolder)
        layout.addWidget(self.btnScan)

        _widget_input = QWidget()
        _widget_input.setLayout(layout)  

        _widget_main = QWidget()
        main = QVBoxLayout(_widget_main)   
        main.addWidget(self._initWidgetDatabase())
        main.addWidget(_widget_input)
        main.addWidget(self._initWidgetLog()) 

        self.setLayout(main) 
        self.setCentralWidget(_widget_main)
        #recalculate
        self.updateFolder("./data")

    def _initWidgetDatabase(self):
        _widget = QWidget()
        layout = QFormLayout()
        _widget.setLayout(layout)

        self.host = QLineEdit(settings.get("VECTORDB", "HOST", fallback="localhost"), _widget)        
        self.port = QLineEdit(settings.get("VECTORDB", "PORT", fallback="6333"), _widget) 
        layout.addRow('Vector Database Host:', self.host)
        layout.addRow('Vector Database Port:', self.port) 
        return _widget

    def _initWidgetLog(self):
        _widget_logs = QWidget()
        _widget_logs.setLayout(QVBoxLayout())

        _log = QLabel("Logs:")
        _widget_logs.layout().addWidget(_log)
        #log box
        self.widget_log = QPlainTextEdit(self)
        self.widget_log.setReadOnly(True) 
        _widget_logs.layout().addWidget(self.widget_log)

        return _widget_logs
    def selectFolder(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setWindowTitle('Select Folder')
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setOption(QFileDialog.ReadOnly, False)

        if folder_dialog.exec_() == QFileDialog.Accepted:
            selected_folders = folder_dialog.selectedFiles()
            print('Selected Folders:', selected_folders)
            self.updateFolder(selected_folders[0])
    def updateFolder(self, folder): 
        self.folder_data = folder
        self.folder.setText(folder)
        self.countImages(folder)

    def scanFolder(self):
        self.btnScan.setEnabled(False)
        self._thread = Thread(target=self._scan_folder, args=())
        self._thread.daemon = True
        self._thread.start()
        
    def _scan_folder(self):
        db_path = self.folder_data
        total = 0

        try:
            # build models once to store them in the memory
            # otherwise, they will be built after cam started and this will cause delays
            DeepFace.build_model(model_name=model_name)
            target_size = functions.find_target_size(model_name=model_name)

            self.client = QdrantClient(self.host.text(), port= int(self.port.text()))

            employees = self._findImages(db_path)

            if len(employees) == 0:
                raise ValueError(
                    "There is no image in ",
                    db_path,
                    " folder! Validate .jpg or .png files exist in this path.",
                )
             
            
            for (employee, id) in employees:
                img_objs = DeepFace.extract_faces(
                    img_path=employee,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    grayscale=False,
                    enforce_detection=True,
                    align=True,
                )

                for img_obj in img_objs:
                    img_region = img_obj["facial_area"]
                    embedding_obj = DeepFace.represent(
                        img_path=img_obj["face"],
                        model_name=model_name,
                        enforce_detection=True,
                        detector_backend="skip",
                        align=True,
                    )

                    img_representation = embedding_obj[0]["embedding"]

                    instance = {
                        "img": employee,
                        "id": id,
                        "face_region": img_region.copy(),
                        "type": "validated",
                        "class": "undefined"
                    }
                    item = {"represent": img_representation, "instance": instance}
                    self.upssert_item(item) 
                    total = total +1
                    self.percent.append(int(total/ len(employees)*100))
        except:
            pass
        print ("total import recode: ", total)
        self.btnScan.setEnabled(True)

    def upssert_item(self, item):
        global collection_name, uuid
        result  = self.client.search(
            collection_name= collection_name, 
            query_vector= item["represent"],
            with_vectors=False,
            with_payload=False,
            limit= 1,
        )
        # print(result)

        if result[0].score > 0.99:            
            return 0

        p = PointStruct(
                id= uuid.uuid4().urn,
                vector= item["represent"],
                payload= item["instance"]
            )

        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[p]
        )
        return 1
    def _findImages(self, db_path):
        employees = []
        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    id = file.lower().split(".")[0]
                    employees.append((exact_path, id))
        return employees
    
    def set_pbar(self):
        self.pbar.setValue(self.percent[-1])
        if self._thread:
            if not self._thread.is_alive():
                self.btnScan.setEnabled(True)
    def countImages(self, folder):
        employees = self._findImages(folder)
        self.btnScan.setText(f'Scan ({len(employees)} Images)')

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()