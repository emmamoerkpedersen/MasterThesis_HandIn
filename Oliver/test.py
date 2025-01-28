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
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle

# Disable Matplotlib picking when zooming
plt.rcParams['toolbar'] = 'toolmanager'


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
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Error selection
        self.error_label = QLabel("Select Error Type:")
        self.layout.addWidget(self.error_label)

        self.error_selector = QComboBox()
        self.error_selector.addItems(["Select Error Type", "Error 1", "Error 2", "Error 3", "Error 4", "Error 5"])
        self.layout.addWidget(self.error_selector)

        # Add span mode selector
        self.span_mode = QComboBox()
        self.span_mode.addItems(["Single Point", "Time Span"])
        self.layout.insertWidget(3, self.span_mode)  # After error selector

        # Buttons
        self.load_button = QPushButton("Load Time Series File")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save Errors")
        self.save_button.clicked.connect(self.save_errors)
        self.layout.addWidget(self.save_button)

        # Table for error logging
        self.error_table = QTableWidget()
        self.error_table.setColumnCount(4)
        self.error_table.setHorizontalHeaderLabels(["Start Time", "End Time", "Error Type", "Duration (s)"])
        self.layout.addWidget(self.error_table)

        # Data attributes
        self.data = None
        self.errors = []
        self.drag_start = None
        self.current_span = None
        self.span_artist = None
        
        # Track if we're in marking mode
        self.marking_mode = True

        # Connect Matplotlib clicks
        self.canvas.mpl_connect("button_press_event", self.on_click)
        # Add key press event to toggle between zoom and marking modes
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add mode indicator
        self.mode_label = QLabel("Current Mode: Error Marking")
        self.layout.insertWidget(0, self.mode_label)  # Add at the top

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
            self.ax.plot(self.data.index, self.data['Value'], label="Time Series Data", picker=False)
            self.ax.set_title("Time Series Visualization")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Value")
            self.ax.legend()
            self.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'm':
            self.toolbar.pan() if self.toolbar.mode == 'pan' else self.toolbar.zoom()
            self.marking_mode = True
            self.mode_label.setText("Current Mode: Error Marking")
        elif event.key == 'z':
            self.marking_mode = False
            self.mode_label.setText("Current Mode: Zoom/Pan")

    def on_click(self, event):
        if not self.marking_mode or self.toolbar.mode != '':
            return
            
        if event.button == 1:  # Left click
            if self.span_mode.currentIndex() == 0:  # Single point
                self.handle_single_click(event)
            else:  # Time span
                self.handle_span_click(event)
                
    def handle_single_click(self, event):
        # Existing single point handling
        timestamp = self.get_timestamp(event)
        if not timestamp:
            return
            
        # Add marker and log error
        self.ax.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
        self.log_error(timestamp, timestamp)  # Duration 0

    def handle_span_click(self, event):
        timestamp = self.get_timestamp(event)
        if not timestamp:
            return
            
        if not self.drag_start:  # First click
            self.drag_start = timestamp
            self.current_span = [timestamp, timestamp]
            self.span_artist = self.ax.axvspan(timestamp, timestamp, 
                                              color='r', alpha=0.3)
            self.canvas.draw()
        else:  # Second click
            self.current_span[1] = timestamp
            self.span_artist.set_xy([[self.current_span[0], 0], 
                                   [self.current_span[0], 1],
                                   [self.current_span[1], 1],
                                   [self.current_span[1], 0]])
            self.log_error(self.current_span[0], self.current_span[1])
            self.drag_start = None
            self.current_span = None
            self.canvas.draw()

    def get_timestamp(self, event):
        if event.xdata is None:
            return None
        return pd.to_datetime(mdates.num2date(event.xdata))

    def log_error(self, start, end):
        selected_error = self.error_selector.currentIndex()
        if selected_error == 0:
            QMessageBox.warning(self, "No Error Selected", "Please select an error type.")
            return
            
        duration = (end - start).total_seconds()
        self.errors.append({
            "start": start,
            "end": end,
            "error_type": selected_error,
            "duration": duration
        })
        self.update_table()

    def update_table(self):
        self.error_table.setRowCount(len(self.errors))
        self.error_table.setColumnCount(4)
        self.error_table.setHorizontalHeaderLabels(
            ["Start Time", "End Time", "Error Type", "Duration (s)"]
        )
        
        for i, error in enumerate(self.errors):
            self.error_table.setItem(i, 0, QTableWidgetItem(str(error["start"])))
            self.error_table.setItem(i, 1, QTableWidgetItem(str(error["end"])))
            self.error_table.setItem(i, 2, QTableWidgetItem(f"Error {error['error_type']}"))
            self.error_table.setItem(i, 3, QTableWidgetItem(f"{error['duration']:.1f}"))

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
