import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt

def load_vst_file(file_path):
    """Load a VST file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, 
                           encoding=encoding,
                           delimiter=';',
                           decimal=',',
                           skiprows=3,
                           names=['Date', 'Value'])

            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    return None

class ErrorLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Error Labeling Tool")
        self.setGeometry(100, 100, 1000, 600)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(NavigationToolbar(self.canvas, self))
        self.layout.addWidget(self.canvas)

        # Error selection
        self.error_label = QLabel("Select Error Type:")
        self.layout.addWidget(self.error_label)

        self.error_selector = QComboBox()
        self.error_selector.addItems(["Select Error Type", "Error 1", "Error 2", "Error 3", "Error 4", "Error 5"])
        self.layout.addWidget(self.error_selector)

        # Buttons
        self.load_button = QPushButton("Load Time Series File")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save Errors")
        self.save_button.clicked.connect(self.save_errors)
        self.layout.addWidget(self.save_button)

        # Table for error logging
        self.error_table = QTableWidget()
        self.error_table.setColumnCount(3)
        self.error_table.setHorizontalHeaderLabels(["Timestamp", "Error Type", "Duration (s)"])
        self.layout.addWidget(self.error_table)

        # Data attributes
        self.data = None
        self.errors = []
        self.last_error_time = None

        # Connect Matplotlib clicks
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Time Series File", "", "Text Files (*.txt)")
        if file_path:
            self.data = load_vst_file(file_path)
            if self.data is not None:
                self.data.set_index('Date', inplace=True)
                self.plot_data()
            else:
                QMessageBox.warning(self, "File Load Error", "Failed to load the file. Please check the format.")

    def plot_data(self):
        if self.data is not None:
            self.ax.clear()
            self.ax.plot(self.data.index, self.data['Value'], label="Time Series Data")
            self.ax.set_title("Time Series Visualization")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Value")
            self.ax.legend()
            self.canvas.draw()

    def on_click(self, event):
        if self.data is None:
            QMessageBox.warning(self, "No Data Loaded", "Please load a time series file first.")
            return

        selected_error = self.error_selector.currentIndex()
        if selected_error == 0:
            QMessageBox.warning(self, "No Error Selected", "Please select an error type before logging.")
            return

        timestamp = pd.Timestamp(event.xdata)
        error_type = selected_error

        # Calculate duration since last error
        duration = None
        if self.last_error_time is not None:
            duration = (timestamp - self.last_error_time).total_seconds()
        self.last_error_time = timestamp

        # Log error
        self.errors.append({"timestamp": timestamp, "error_type": error_type, "duration": duration})
        self.update_table()

    def update_table(self):
        self.error_table.setRowCount(len(self.errors))
        for i, error in enumerate(self.errors):
            self.error_table.setItem(i, 0, QTableWidgetItem(str(error["timestamp"])))
            self.error_table.setItem(i, 1, QTableWidgetItem(f"Error {error["error_type"]}"))
            self.error_table.setItem(i, 2, QTableWidgetItem(str(error["duration"]) if error["duration"] else "N/A"))

    def save_errors(self):
        if not self.errors:
            QMessageBox.warning(self, "No Errors Logged", "There are no errors to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Errors", "", "CSV Files (*.csv)")
        if file_path:
            error_df = pd.DataFrame(self.errors)
            error_df.to_csv(file_path, index=False)
            QMessageBox.information(self, "File Saved", f"Errors saved to {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ErrorLabelingApp()
    window.show()
    sys.exit(app.exec())
