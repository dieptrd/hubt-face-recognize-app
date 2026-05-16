import time
from PyQt5.QtWidgets import QApplication

def spin(seconds):
    """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
    time_end = time.time() + seconds
    while time.time() < time_end:
        QApplication.processEvents()
        
def _safe_get(obj, *keys, default=None):
    try:
        cur = obj
        for key in keys:
            if cur is not None and isinstance(cur, dict):
                cur = cur.get(key, None)
            elif hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                return default
        return cur
    except Exception as e:
        print("Error in _safe_get: ", e)
        return default