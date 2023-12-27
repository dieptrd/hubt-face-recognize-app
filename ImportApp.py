from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
import sys 
from collections import deque
import os 
from appSettings import settings
from threading import Thread
import time
import uuid
import numpy as np
import cv2
from PIL import Image
import base64
import json

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from deepface import DeepFace
from deepface.commons import functions

model_name = "VGG-Face"
detector_backend = "retinaface"
vector_size = 2622
collection_name="hubt_faces"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
 
# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._thread = None 
        self.percent = deque([0], maxlen=1)
        self.import_image_queue = deque(maxlen=1)
        self.logs = deque(maxlen=10)
        self.setWindowTitle("Add Images to database") 
        
        
        self.pbar = QProgressBar(self)    

        _widget_main = QWidget(self)
        main = QVBoxLayout(_widget_main)   
        main.addWidget(self._initWidgetDatabase()) 
        main.addWidget(self._initWidgetFolder())
        main.addWidget(self.pbar)
        main.addWidget(self._initAction())
        #read img
        self.import_image = QLabel("", self)
        main.addWidget(self.import_image)
        #log box
        main.addWidget(self._initWidgetLog()) 

        self.setLayout(main) 
        self.setCentralWidget(_widget_main)
        #recalculate
        self.updateFolder("./data")  

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_UI)
        self.timer.start(1)

    def _initAction(self):
        _widget = QWidget()
        _widget.setLayout(QHBoxLayout())
        _widget.layout().addWidget(QLabel(""))
        
        btnClear = QPushButton(self)
        btnClear.setText("Clear Data")
        btnClear.clicked.connect(self.clearDatabase) 

        self.btnScan = QPushButton(self)
        self.btnScan.setText("Import (0 Images)")
        self.btnScan.clicked.connect(self.scanFolder)

        _widget.layout().addWidget(btnClear)
        _widget.layout().addWidget(self.btnScan)
        return _widget
    
    def _initWidgetFolder(self):
        _openFolder = QPushButton(self)
        _openFolder.setText("...")
        _openFolder.clicked.connect(self.selectFolder)
        self.folder = QLineEdit("", self)
        self.folder.setReadOnly(True) 
        
        _widget = QWidget()
        _widget.setLayout(QHBoxLayout())
        _widget.layout().addWidget(QLabel("Faces Image's Folder:"))
        _widget.layout().addWidget(self.folder)
        _widget.layout().addWidget(_openFolder)
        return _widget

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
        self.addLog("Folder data changed.") 
        self.folder.setText(folder)
        self.countImages(folder)

    def scanFolder(self):
        self.btnScan.setEnabled(False)
        self._thread = Thread(target=self._scan_folder, args=())
        self._thread.daemon = True
        self._thread.start()

    def clearDatabase(self):
        global collection_name
        self.addLog("==Clear Vectors Database==")
        client = None
        try:
            client = QdrantClient(self.host.text(), port= int(self.port.text()))
            collections = client.get_collections()
            exists = False
            for _, c in collections:
                if c[0].name == collection_name:
                   exists = True
                   break
            if exists:
                info = client.get_collection(collection_name=collection_name)
                dialog = QMessageBox(self)
                dialog.setIcon(QMessageBox.Warning)
                dialog.setWindowTitle("Warning")
                dialog.setText(f'DB had {info.points_count} vectors')
                dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel) 
                if dialog.exec() == QMessageBox.Ok:
                    if client.delete_collection(collection_name=collection_name):
                        #delete success 
                        self.addLog("DB deleted")
                else:
                    raise ValueError("User Cancel.")
            #create new
            client.create_collection(
                collection_name= collection_name,
                vectors_config= VectorParams(size=vector_size, distance=Distance.COSINE),
            )    
        except Exception as ex:    
            print(ex)
            self.addLog(f'{ex}')
            pass

        self.addLog("==Clear Vectors Database Done==")
        return 1

    def addLog(self, msg):
        t = time.time()
        self.logs.append((time,msg))
        
    def _scan_folder(self):
        db_path = self.folder.text()
        total = 0
        
        self.addLog("===Scan images for import.===")
        try:
            # build models once to store them in the memory
            # otherwise, they will be built after cam started and this will cause delays
            DeepFace.build_model(model_name=model_name)
            target_size = functions.find_target_size(model_name=model_name)

            client = QdrantClient(self.host.text(), port= int(self.port.text()))

            employees = self._findImages(db_path)

            if len(employees) == 0:
                raise ValueError(
                    "There is no image in ",
                    db_path,
                    " folder! Validate .jpg or .png files exist in this path.",
                )
             
            
            for (employee, id, ext) in employees:
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
                    img, _ = functions.load_image(employee)
                    img_base64 = f'data:image/{ext};base64,' + base64.b64encode(cv2.imencode(ext, img)[1]).decode()
                    self.import_image_queue.append(img_base64)
                    instance = {
                        "img": employee,
                        "base64": img_base64,
                        "id": id,
                        "face_region": img_region.copy(),
                        "type": "validated",
                        "class": "undefined"
                    }
                    item = {"represent": img_representation, "instance": instance}
                    self.upssert_item(item, client) 
                    total = total +1
                    self.percent.append(int(total/ len(employees)*100))
            
        except Exception as ex: 
            self.addLog(f'scan error: {ex}')
            pass
        self.addLog(f'total import recode: {total}')
        self.addLog("===Scan End===")
    
    def upssert_item(self, item, client):
        global collection_name, uuid
        result  = client.search(
            collection_name= collection_name, 
            query_vector= item["represent"],
            with_vectors=False,
            with_payload=False,
            limit= 1,    
            score_threshold= 0.9       
        ) 

        if len(result) > 0:
            if result[0].score > 0.99:            
                return 0

        p = PointStruct(
                id= uuid.uuid4().urn,
                vector= item["represent"],
                payload= item["instance"]
            )

        client.upsert(
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
                    segment = file.lower().split(".")
                    employees.append((exact_path, segment[0], f'.{segment[1]}'))
        return employees
    
    def update_UI(self):
        if len(self.import_image_queue) > 0:
            img, _ = functions.load_image(self.import_image_queue.pop())
            pix = self.convert_cv_qt(img)
            self.import_image.setPixmap(pix)

        self.pbar.setValue(self.percent[-1])
        while len(self.logs) > 0:
            time, msg = self.logs.popleft()
            self.widget_log.moveCursor(QtGui.QTextCursor.Start)
            self.widget_log.insertPlainText(f'{time.strftime("%H.%M.%S")} : ' + msg + "\r\n")
        if self._thread:
            if not self._thread.is_alive():
                self.btnScan.setEnabled(True)
    def countImages(self, folder):
        employees = self._findImages(folder)
        self.btnScan.setText(f'Scan ({len(employees)} Images)') 
    
    def _rever_image(self, img):
        if(img.max() < 1):
            a = img.copy()
            img = np.interp(a, (a.min(), a.max()), (0,255)).astype(np.uint8) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb image
        return img
    
    def convert_cv_qt(self, img):
        cv_img = self._rever_image(img)

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(cv_img.shape[0], cv_img.shape[1], QtCore.Qt.AspectRatioMode.KeepAspectRatio)        
        return QtGui.QPixmap.fromImage(p)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()