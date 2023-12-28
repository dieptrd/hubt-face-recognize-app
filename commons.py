import time
from PyQt5.QtWidgets import QApplication

def spin(seconds):
    """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
    time_end = time.time() + seconds
    while time.time() < time_end:
        QApplication.processEvents()