
# Joe Hollowed
# University of Michigan 2023
# 
# This class provides methods for reading and reducing zonally averaged datasets, and communicating results to callers from the GUI

import os
import numpy as np
import xarray as xr
from PyQt5.QtWidgets import QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from util import pyqt_set_trace as set_trace
from util import time2day, raise_error, clear_layout


# ==================================================================


# ---- global vars
MASTER_VAR_LIST = ['SO2', 'SULFATE', 'AOD', 'T025', 'T050', 'T1000']
MASTER_COORD_LIST = ['lat', 'lev', 'time']

# ---- for updating progress bar
PYFILE = os.path.abspath(__file__)
p_increment = 100 / (sum(1 for line in open(os.path.abspath(__file__)) if 'incr_pbar(self.pbar)' in line) - 1)
def incr_pbar(pbar): 
    pbar.setProperty("value", pbar.value() + p_increment)
    QApplication.processEvents()


# ==================================================================


class data_plotter:
    def __init__(self, var, band, data_handler, viewport, pbar, global_scale):
        '''
        Class containing methods for plotting data from an associated data_handler in the GUI plotPanel
        
        Parameters
        ----------
        var : str
            name of variable to plot
        band : int
            latitude band to plot, as an integer index of data_handler.data_avg_bands
        data_handller : data_handler object
            handle to a data_handler object which has previously read in and analyzed the data
        viewport : QGraphicsView
            handle to the GUI plot viewport as a QGraphicsView object
        pbar : QProgressBar object
            handle to the progress bar object
        global_scale : bool
            whether or not to scale the y-axis of all SAI variable figures to a common range
        '''

        self.var  = var
        self.band = band
        self.data_handler = data_handler
        self.viewport = viewport
        self.pbar = pbar
        self.global_scale = global_scale
      
        # ---- clear the current layout if exists
        if(self.viewport.layout() is None):
            layout = QVBoxLayout()
        else:
            clear_layout(self.viewport.layout())
            layout = self.viewport.layout()

        # ---- build new layout with matplotlib toolbar and figure
        self.viewport.figure  = plt.Figure()
        self.viewport.canvas  = FigureCanvas(self.viewport.figure)
        self.viewport.toolbar = NavigationToolbar(self.viewport.canvas, self.viewport)
        layout.addWidget(self.viewport.toolbar)
        layout.addWidget(self.viewport.canvas)
        self.viewport.setLayout(layout)
        incr_pbar(self.pbar)

        # ---- get data
        self.time = time2day(self.data_handler.data_avg_bands[self.band]['time'])
        self.var_data  = self.data_handler.data_avg_bands[self.band][self.var]
        self.anom_base_data = -(self.data_handler.anom_bands[self.band][self.var] - self.var_data)
        incr_pbar(self.pbar)

        # ---- get data limits
        self.datalim = [self.data_handler.avg_bands_min[var], self.data_handler.avg_bands_max[var]]
        self.datadel = abs(self.datalim[0] - self.datalim[1])

    # ==================================================================

    def plot_data(self):
        '''
        Plots the requested data and formats the figure
        '''
        self.ax = self.viewport.figure.add_subplot(111)
        self.ax.plot(self.time, self.var_data, '-r')
        self.ax.plot(self.time, self.anom_base_data, '-k')
        if(self.var[0] != 'T' and self.global_scale):
            self.ax.set_xlim(min(self.time), max(self.time))
            self.ax.set_ylim(self.datalim[0] - self.datadel*0.05, 
                             self.datalim[1] + self.datadel*0.05)
        self.ax.grid()
        self.viewport.canvas.draw()
        incr_pbar(self.pbar)


# ==================================================================







