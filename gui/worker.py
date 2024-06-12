from PyQt5.QtCore import QThread, pyqtSignal

class Worker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    info = pyqtSignal(str)
    progress = pyqtSignal(int) 
    stop_signal = pyqtSignal()

    def __init__(self, func, ticker=None, start_date=None, end_date=None):
        super().__init__()
        self.func = func
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self._is_running = True
        self.stop_signal.connect(self.stop)

    def run(self):
        try:
            if self.ticker:
                self.info.emit(f"Starting {self.func.__name__} for {self.ticker}")
                if 'start_date' in self.func.__code__.co_varnames and 'end_date' in self.func.__code__.co_varnames:
                    self.func(self.ticker, start_date=self.start_date, end_date=self.end_date, worker=self)
                else:
                    self.func(self.ticker, worker=self)
            else:
                self.info.emit(f"Starting {self.func.__name__}")
                self.func(worker=self)
            if self._is_running:
                self.info.emit(f"{self.func.__name__} completed")
                self.finished.emit()
            else:
                self.info.emit(f"{self.func.__name__} stopped")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False
        self.info.emit("Stop signal received")

    def update_progress(self, value):
        self.progress.emit(value)
