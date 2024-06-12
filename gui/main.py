import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QTextEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
import logging
from datetime import datetime, timedelta

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

from src.train import main as train_main
from src.data_preparation import main as data_preparation_main
from src.feature_engineering import main as feature_engineering_main
from src.predict import main as predict_main

class Worker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    info = pyqtSignal(str)
    stop_signal = pyqtSignal()

    def __init__(self, func, ticker=None, start_date=None, end_date=None):
        super().__init__()
        self.func = func
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self._is_running = True

    def run(self):
        try:
            if self.ticker:
                self.info.emit(f"Starting {self.func.__name__} for {self.ticker}")
                self.func(self.ticker, start_date=self.start_date, end_date=self.end_date, worker=self)
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
        if not self._is_running:
            return
        self._is_running = False
        self.info.emit("Stop signal received")

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('gui/main_window.ui', self)

        # Connect UI elements
        self.trainButton = self.findChild(QPushButton, 'trainButton')
        self.stopButton = self.findChild(QPushButton, 'stopButton')
        self.dataPreparationButton = self.findChild(QPushButton, 'dataPreparationButton')
        self.featureEngineeringButton = self.findChild(QPushButton, 'featureEngineeringButton')
        self.predictButton = self.findChild(QPushButton, 'predictButton')
        self.tickerInput = self.findChild(QLineEdit, 'tickerInput')
        self.startDateInput = self.findChild(QLineEdit, 'startDateInput')
        self.endDateInput = self.findChild(QLineEdit, 'endDateInput')
        self.logOutput = self.findChild(QTextEdit, 'logOutput')

        # Initialize stop button as hidden
        self.stopButton.hide()

        # Set default dates
        self.set_default_dates()

        # Connect the buttons to the functions
        self.trainButton.clicked.connect(self.start_training)
        self.stopButton.clicked.connect(self.stop_training)
        self.dataPreparationButton.clicked.connect(self.start_data_preparation)
        self.featureEngineeringButton.clicked.connect(self.start_feature_engineering)
        self.predictButton.clicked.connect(self.start_prediction)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        handler = QTextEditLogger(self.logOutput)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)

        self.worker = None
        self.data_prepared = False
        self.features_engineered = False

    def set_default_dates(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        self.startDateInput.setText(start_date)
        self.endDateInput.setText(end_date)

    def start_training(self):
        if not self.data_prepared or not self.features_engineered:
            self.logger.error("Data preparation and feature engineering must be completed before training.")
            return

        ticker = self.tickerInput.text()
        self.worker = Worker(train_main, ticker)
        self.worker.info.connect(self.logger.info)
        self.worker.error.connect(self.logger.error)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

        # Show stop button and hide other buttons
        self.stopButton.show()
        self.toggle_buttons(False)

    def start_data_preparation(self):
        ticker = self.tickerInput.text()
        start_date = self.startDateInput.text()
        end_date = self.endDateInput.text()
        self.worker = Worker(data_preparation_main, ticker, start_date, end_date)
        self.worker.info.connect(self.logger.info)
        self.worker.error.connect(self.logger.error)
        self.worker.finished.connect(self.on_data_preparation_finished)
        self.worker.start()

        # Show stop button and hide other buttons
        self.stopButton.show()
        self.toggle_buttons(False)

    def start_feature_engineering(self):
        if not self.data_prepared:
            self.logger.error("Data preparation must be completed before feature engineering.")
            return

        ticker = self.tickerInput.text()
        self.worker = Worker(feature_engineering_main, ticker)
        self.worker.info.connect(self.logger.info)
        self.worker.error.connect(self.logger.error)
        self.worker.finished.connect(self.on_feature_engineering_finished)
        self.worker.start()

        # Show stop button and hide other buttons
        self.stopButton.show()
        self.toggle_buttons(False)

    def start_prediction(self):
        ticker = self.tickerInput.text()
        self.worker = Worker(predict_main, ticker)
        self.worker.info.connect(self.logger.info)
        self.worker.error.connect(self.logger.error)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

        # Show stop button and hide other buttons
        self.stopButton.show()
        self.toggle_buttons(False)

    def stop_training(self):
        if self.worker is not None:
            self.worker.stop_signal.emit()
            self.worker.wait()  # Ensure the thread has finished before continuing

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop_signal.emit()
            self.worker.wait()  # Wait for the thread to finish before closing
        event.accept()

    def on_process_finished(self):
        self.logger.info("Process finished.")
        self.worker = None

        # Show other buttons and hide stop button
        self.stopButton.hide()
        self.toggle_buttons(True)

    def on_data_preparation_finished(self):
        self.logger.info("Data preparation finished.")
        self.data_prepared = True
        self.on_process_finished()

    def on_feature_engineering_finished(self):
        self.logger.info("Feature engineering finished.")
        self.features_engineered = True
        self.on_process_finished()

    def toggle_buttons(self, enabled):
        self.dataPreparationButton.setEnabled(enabled)
        self.featureEngineeringButton.setEnabled(enabled and self.data_prepared)
        self.trainButton.setEnabled(enabled and self.data_prepared and self.features_engineered)
        self.predictButton.setEnabled(enabled)

class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        self.text_edit.append(msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
