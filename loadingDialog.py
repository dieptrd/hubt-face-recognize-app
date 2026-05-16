from PyQt5 import QtWidgets, QtCore


class LoadingDialog(QtWidgets.QDialog):
    """Simple modal loading dialog with a label and progress bar.

    Usage:
        dlg = LoadingDialog(parent)
        dlg.setRange(0, 1000)
        dlg.setValue(0)
        dlg.setLabelText("Loading models...")
        dlg.show()  # or dlg.exec_() for blocking modal

    Thread-safety: use the provided slots or emit signals connected to
    `setValue` and `setLabelText` from worker threads.
    """

    def __init__(self, parent=None, title="Loading", minimum=0, maximum=1000):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setModal(True)
        self.setMinimumSize(360, 120)

        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)
        self.label.setText("")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(minimum, maximum)
        self.progress.setValue(minimum)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # optional detail area
        self.detail = QtWidgets.QLabel(self)
        self.detail.setText("")
        self.detail.setWordWrap(True)
        self.detail.setVisible(False)
        layout.addWidget(self.detail)

        # disable window close by default; you can enable with allow_cancel()
        self._allow_cancel = False
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)

    # -- thread-safe slots --
    @QtCore.pyqtSlot(int)
    def setValue(self, value: int):
        try:
            self.progress.setValue(int(value))
        except Exception:
            pass

    @QtCore.pyqtSlot(int, int)
    def setRange(self, minimum: int, maximum: int):
        try:
            self.progress.setRange(int(minimum), int(maximum))
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def setLabelText(self, text: str):
        try:
            self.label.setText(str(text))
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def setDetailText(self, text: str):
        try:
            self.detail.setText(str(text))
            self.detail.setVisible(bool(text))
        except Exception:
            pass

    def allow_cancel(self, allow: bool = True):
        """Enable or disable the window close button (and cancel behavior).

        If enabled, the dialog will show a close button and emit rejected() when
        closed by the user.
        """
        self._allow_cancel = bool(allow)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, self._allow_cancel)
        self.setModal(not self._allow_cancel)

    # convenience methods to connect signals from worker threads
    def bind_signals(self, worker):
        """Bind common worker signals to the dialog.

        Expected worker signals (optional): progress(int,str), finished(), detail(str), value(int).
        This method attempts to connect the signals if present.
        """
        try:
            if hasattr(worker, 'progress'):
                # progress may be (int, str) or int
                try:
                    worker.progress.connect(lambda v, l=None: (self.setValue(v), self.setLabelText(l) if l else None))
                except Exception:
                    try:
                        worker.progress.connect(self.setValue)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if hasattr(worker, 'detail'):
                worker.detail.connect(self.setDetailText)
        except Exception:
            pass

        try:
            if hasattr(worker, 'value'):
                worker.value.connect(self.setValue)
        except Exception:
            pass
        self.worker = worker  # keep reference to prevent GC

    # small helper to run the dialog in a blocking fashion while a QThread worker runs
    def exec_with_thread(self, timeout: int = 0):
        """Start a QThread (if not started), show the dialog modally and block until
        the thread finishes or an optional timeout (ms) elapses. Returns True if the
        thread finished, False otherwise.
        """
        finished = False
        # create a non-parented thread so it's not destroyed when the dialog/widget
        # is closed; keep a reference to avoid GC
        thread = QtCore.QThread()
        self._thread = thread

        def on_thread_finished():
            nonlocal finished
            finished = True
            try:
                if self.isVisible():
                    self.close()
            except Exception:
                pass

        thread.finished.connect(on_thread_finished)

        # move worker to the thread and start its `run` when thread starts
        self.worker.moveToThread(thread)
        if hasattr(self.worker, 'run'):
            try:
                thread.started.connect(self.worker.run)
            except Exception:
                pass

        # if worker emits finished, stop and cleanup the thread
        if hasattr(self.worker, 'finished'):
            try:
                self.worker.finished.connect(thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
            except Exception:
                pass

        thread.finished.connect(thread.deleteLater)
        thread.start()

        # run modal loop until thread finishes or timeout
        if timeout and timeout > 0:
            loop = QtCore.QEventLoop()
            # thread will call on_thread_finished which closes the dialog
            # we quit the loop when the dialog closes
            self.finished.connect(loop.quit)
            QtCore.QTimer.singleShot(timeout, loop.quit)
            self.exec_()
            loop.exec()
        else:
            self.exec_()

        return finished
