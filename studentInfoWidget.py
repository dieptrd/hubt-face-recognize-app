import pyodbc
from PyQt5 import QtWidgets, QtCore

class StudentInfoWidget(QtWidgets.QWidget):
    def __init__(self, access_db_path, parent=None):
        super().__init__(parent)
        self.access_db_path = access_db_path
        self.init_ui()
        self.conn = None

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.info_label = QtWidgets.QLabel("Thông tin sinh viên sẽ hiển thị ở đây")
        self.info_label.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(self.info_label)

    def connect_db(self):
        if self.conn is None:
            conn_str = (
                r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                f'DBQ={self.access_db_path};'
            )
            self.conn = pyodbc.connect(conn_str)

    def show_student_info(self, msv):
        try:
            self.connect_db()
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM Students WHERE MSV = ?", (msv,))
            row = cursor.fetchone()
            if row:
                # Giả sử các trường: MSV, Name, Class, Birthday
                info = f"MSV: {row.MSV}\nTên: {row.Name}\nLớp: {row.Class}\nNgày sinh: {row.Birthday}"
            else:
                info = "Không tìm thấy sinh viên"
            self.info_label.setText(info)
        except Exception as e:
            self.info_label.setText(f"Lỗi truy vấn: {e}")