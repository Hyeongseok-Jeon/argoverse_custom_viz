import PyQt5.QtWidgets
from PyQt5 import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, PyQt5.QtWidgets.QSizePolicy.Expanding,PyQt5.QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class matplotlibWidget(PyQt5.QtWidgets.QWidget):

    def __init__(self, parent = None):
        PyQt5.QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = PyQt5.QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)