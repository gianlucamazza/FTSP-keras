# qtext_edit_logger.py
import logging
from PyQt5.QtCore import pyqtSignal, QObject

class LogEmitter(QObject):
    log_signal = pyqtSignal(str)

class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.emitter = LogEmitter()
        self.emitter.log_signal.connect(self.append_log)

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_signal.emit(msg)

    def append_log(self, msg):
        self.text_edit.append(msg)
