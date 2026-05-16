from PyQt5 import QtWidgets, QtGui, QtCore
import imutils
import cv2
import numpy as np

class FaceCompareWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, show_info=False):
        # keep constructor compatible with calls that pass parent as first positional
        super().__init__(parent)
        self.show_info = show_info
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
        if self.show_info:
            info_layout = QtWidgets.QVBoxLayout()
            self.msv_label = QtWidgets.QLabel("MSV: ")
            self.name_label = QtWidgets.QLabel("Tên: ")
            info_layout.addWidget(self.msv_label)
            info_layout.addWidget(self.name_label)
            main_layout.addLayout(info_layout)

    def set_camera_face(self, pix):
        try:
            # numpy.ndarray (OpenCV frame, usually BGR)
            if isinstance(pix, np.ndarray):
                img = pix
                if img.size == 0:
                    self.camera_face_label.clear()
                    return

                # Convert BGR -> RGB for display
                if img.ndim == 3 and img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.ndim == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = img

                # Resize to label width while keeping aspect ratio
                w_label = self.camera_face_label.width()
                img_resized = imutils.resize(img_rgb, width=w_label)

                h, w = img_resized.shape[:2]
                bytes_per_line = 3 * w
                qimg = QtGui.QImage(img_resized.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pix_face = QtGui.QPixmap.fromImage(qimg)
                self.camera_face_label.setPixmap(pix_face.scaled(w_label, self.camera_face_label.height(), QtCore.Qt.KeepAspectRatio))
                return

            # QImage
            if isinstance(pix, QtGui.QImage):
                pix_face = QtGui.QPixmap.fromImage(pix)
                self.camera_face_label.setPixmap(pix_face.scaled(200, 200, QtCore.Qt.KeepAspectRatio))
                return

            # QPixmap
            if isinstance(pix, QtGui.QPixmap):
                self.camera_face_label.setPixmap(pix.scaled(200, 200, QtCore.Qt.KeepAspectRatio))
                return

            # Fallback: try to construct QPixmap
            qpix = QtGui.QPixmap(pix)
            self.camera_face_label.setPixmap(qpix.scaled(200, 200, QtCore.Qt.KeepAspectRatio))
        except Exception as e:
            print("set_camera_face error:", e)
            self.camera_face_label.clear()

    # def set_db_face(self, qimage):
    #     pix = QtGui.QPixmap.fromImage(qimage)
    #     self.db_face_label.setPixmap(pix.scaled(200, 200, QtCore.Qt.KeepAspectRatio))

    def set_info(self, msv, name):
        if self.show_info:
            self.msv_label.setText(f"MSV: {msv}")
            self.name_label.setText(f"Tên: {name}")
