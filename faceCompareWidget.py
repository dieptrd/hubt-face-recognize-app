from PyQt5 import QtWidgets, QtGui, QtCore

class FaceCompareWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        
        main_layout.addWidget(QtWidgets.QLabel("Xác định khuôn mặt"))
        # HBox cho 2 ảnh trên cùng một dòng
        images_layout = QtWidgets.QHBoxLayout()
        self.camera_face_label = QtWidgets.QLabel("Camera Face")
        self.camera_face_label.setFixedSize(200, 200)
        self.camera_face_label.setStyleSheet("border: 1px solid gray;")
        images_layout.addWidget(self.camera_face_label)

        # self.db_face_label = QtWidgets.QLabel("Database Face")
        # self.db_face_label.setFixedSize(200, 200)
        # self.db_face_label.setStyleSheet("border: 1px solid gray;")
        # images_layout.addWidget(self.db_face_label)

        main_layout.addLayout(images_layout)

        # Thông tin sinh viên nằm dưới
        info_layout = QtWidgets.QVBoxLayout()
        self.msv_label = QtWidgets.QLabel("MSV: ")
        self.name_label = QtWidgets.QLabel("Tên: ")
        info_layout.addWidget(self.msv_label)
        info_layout.addWidget(self.name_label)
        main_layout.addLayout(info_layout)

    def set_camera_face(self, pix):
        self.camera_face_label.setPixmap(pix.scaled(200, 200, QtCore.Qt.KeepAspectRatio))

    # def set_db_face(self, qimage):
    #     pix = QtGui.QPixmap.fromImage(qimage)
    #     self.db_face_label.setPixmap(pix.scaled(200, 200, QtCore.Qt.KeepAspectRatio))

    def set_info(self, msv, name):
        self.msv_label.setText(f"MSV: {msv}")
        self.name_label.setText(f"Tên: {name}")