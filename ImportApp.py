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
        
        self.configs = {
            "model_name": "VGG-Face",
            "distance_metric": "cosine",
            "detector_backend": "retinaface",
            "enforce_detection": True,
            "target_size": (224, 224),
            "collection_name": "hubt_faces",
            "vector_size": 2622,
            "home_folder": settings.get_deepface_home(),
            "images_folder": "./data",
            "db_host": settings.get("VECTORDB", "HOST", fallback="localhost"),
            "db_port": settings.get("VECTORDB", "PORT", fallback="6333"),
        }
        
        self.pbar = QProgressBar(self)    

        _widget_main = QWidget(self)
        main = QVBoxLayout(_widget_main)   
        main.addWidget(self._initWidgetDatabase()) 
        main.addWidget(self._init_DeepfaceWidget())
        main.addWidget(self._initWidgetImageFolder())
        main.addWidget(self.pbar)
        main.addWidget(self._initAction())
        #read img
        self.import_image = QLabel("", self)
        self.import_image.setAlignment(QtCore.Qt.AlignCenter)
        main.addWidget(self.import_image)
        #log box
        main.addWidget(self._initWidgetLog()) 

        _widget_main.setLayout(main) 
        self.setCentralWidget(_widget_main)
        #recalculate
        self.updateImageFolder("./data")  

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_UI)
        self.timer.start(1)

    def update_configs(self, key, value):
        self.configs[key] = value

    def _init_DeepfaceWidget(self):
        _widget = QGroupBox("AI Emberdding Methods", self)
        layout = QFormLayout()
        _widget.setLayout(layout)

        layout.addRow("Home's Folder:", self._initWidgetHomeFolder())

        self.recognize = QComboBox(self)
        self.recognize.addItems("VGG-Face".split(", ")) # VGG-Face, Facenet, OpenFace, DeepFace, DeepID
        self.recognize.setCurrentText(self.configs.get("model_name"))
        self.recognize.currentIndexChanged.connect(lambda: self.update_configs("model_name", self.recognize.currentText()))
        layout.addRow('Recognize Methods', self.recognize)

        return _widget 
        

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
    
    def _initWidgetHomeFolder(self):
        _openFolder = QPushButton(self)
        _openFolder.setText("...")
        _openFolder.clicked.connect(lambda: self.selectHomeFolder(home))
        home = QLineEdit(settings.get_deepface_home(), self)
        home.setReadOnly(True) 
        
        _widget = QWidget()
        _widget.setLayout(QHBoxLayout())
        _widget.layout().setContentsMargins(0,0,0,0)
        _widget.layout().addWidget(home)
        _widget.layout().addWidget(_openFolder)
        return _widget

    def _initWidgetImageFolder(self):
        _openFolder = QPushButton(self)
        _openFolder.setText("...")
        _openFolder.clicked.connect(self.selectImageFolder)
        self.folder = QLineEdit("./data", self)
        self.folder.setReadOnly(True) 
        
        _widget = QWidget()
        _widget.setLayout(QHBoxLayout())
        _widget.layout().addWidget(QLabel("Faces Image's Folder:"))
        _widget.layout().addWidget(self.folder)
        _widget.layout().addWidget(_openFolder)
        return _widget

    def _initWidgetDatabase(self):
        _widget = QGroupBox("Vector Database Service", self)
        layout = QFormLayout()
        _widget.setLayout(layout)

        self.host = QLineEdit(self.configs.get("db_host"), _widget)
        self.host.textChanged.connect(lambda: self.update_configs("db_host", self.host.text()))
        self.port = QLineEdit(self.configs.get("db_port"), _widget) 
        self.port.textChanged.connect(lambda: self.update_configs("db_port", self.port.text()))
        layout.addRow('Host:', self.host)
        layout.addRow('Port:', self.port) 
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
    
    def selectHomeFolder(self, home):
        folder_dialog = QFileDialog(self)
        folder_dialog.setWindowTitle('Select Folder')
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setOption(QFileDialog.ReadOnly, False)
        folder_dialog.setDirectory(settings.get_deepface_home())

        if folder_dialog.exec_() == QFileDialog.Accepted:
            selected_folders = folder_dialog.selectedFiles() 
            settings.set_deepface_home(selected_folders[0])
            self.update_configs("home_folder", selected_folders[0])
            home.setText(selected_folders[0])

    def selectImageFolder(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setWindowTitle('Select Folder')
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setOption(QFileDialog.ReadOnly, False)

        if folder_dialog.exec_() == QFileDialog.Accepted:
            selected_folders = folder_dialog.selectedFiles()
            print('Selected Folders:', selected_folders)
            self.updateImageFolder(selected_folders[0])
    def updateImageFolder(self, folder): 
        self.addLog("Folder data changed.") 
        self.configs["images_folder"] = folder
        self.folder.setText(folder)
        self.countImages(folder)

    
    def scanFolder(self):
        """
        Scans the specified image folder for importing images into the database.
        This method disables the 'Scan' button and starts a new thread to execute the '_scan_folder' method.
        The '_scan_folder' method performs the actual scanning and importing of images.

        Parameters:
            None

        Returns:
            None
        """
        self.btnScan.setEnabled(False)
        self._thread = Thread(target=self._scan_folder, args=())
        self._thread.daemon = True
        self._thread.start()

    def clearDatabase(self):
        collection_name = self.configs.get("collection_name")
        vector_size = self.configs.get("vector_size")
        self.addLog("==Clear Vectors Database==")
        client = None
        try:
            client = QdrantClient(self.configs.get("db_host"), port= int(self.configs.get("db_port"))) 
            exists = self.total_vectors(client) 
            if exists >= 0: 
                dialog = QMessageBox(self)
                dialog.setIcon(QMessageBox.Warning)
                dialog.setWindowTitle("Warning")
                dialog.setText(f'DB had {exists} vectors')
                dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel) 
                if dialog.exec() == QMessageBox.Ok:
                    if client.recreate_collection(
                        collection_name= collection_name,
                        vectors_config= VectorParams(size=vector_size, distance=Distance.COSINE),
                    ):
                        #delete success 
                        self.addLog("DB clear success!")
                else:
                    raise ValueError("User Cancel.")
            #create new
            else:
                self.addLog(f'Create new collection, vector_size is: {vector_size}')
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
    
    def total_vectors(self, conn):
        collection_name = self.configs.get("collection_name")
        try:
            info = conn.get_collection(collection_name=collection_name)
            return info.points_count
        except Exception as e:
            self.addLog("Collection collection_name not exists")
        return -1

    def addLog(self, msg):
        t = time.time()
        self.logs.append((time,msg))
        
    def _scan_folder(self):
        total = 0
        db_path = self.configs.get("images_folder")
        model_name = self.configs.get("model_name")
        detector_backend = self.configs.get("detector_backend")
        self.addLog("===Scan images for import.===")
        try:
            # build models once to store them in the memory
            # otherwise, they will be built after cam started and this will cause delays
            DeepFace.build_model(model_name=self.configs.get("model_name"))
            target_size = functions.find_target_size(model_name=model_name)

            client = QdrantClient(self.configs.get("db_host"), port= int(self.configs.get("db_port")))

            employees = self._findImages(db_path)

            if len(employees) == 0:
                raise ValueError(
                    "There is no image in ",
                    db_path,
                    " folder! Validate .jpg or .png files exist in this path.",
                )
             
            
            for (employee, id, ext, class_name) in employees:
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
                        "img": img_base64,
                        "id": id,
                        "face_region": img_region.copy(),
                        "type": "validated",
                        "class": class_name
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
        global uuid
        collection_name = self.configs.get("collection_name")
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
            segments = r.lower().split(os.sep)
            class_name = segments[-1]
            if segments[-1] == db_path:
                class_name = "undefined"
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    ext_segments = file.lower().split(".")
                    employees.append((exact_path, ext_segments[0], f'.{ext_segments[1]}', class_name))
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