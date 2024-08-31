
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
VAR_COLOR = {'SO2':'limegreen', 'SULFATE':'dodgerblue', 'AOD':'fuchsia', 
                    'T025':'salmon', 'T050':'red', 'T1000':'orange'}
VAR_UNIT = {'SO2':'kg/kg', 'SULFATE':'kg/kg', 'AOD':None, 
                    'T025':'K', 'T050':'K', 'T1000':'K'}
VAR_NAME = {'SO2':'SO2 mixing-ratio', 'SULFATE':'Sulfate mixing-ratio', 'AOD':'Aerosol Optical Depth', 
            'T025':'Stratospheric Temperature at 25 hPa', 'T050':'Stratospheric Temperature at 50 hPa', 
            'T1000':'Surface Temperature at 1000 hPa'}

LABEL_FS   = 14
TITLE_FS = 12
TICK_FS  = 10
plt.rc('font', size=TITLE_FS)        # controls default text sizes
plt.rc('axes', titlesize=TITLE_FS)   # fontsize of the axes title
plt.rc('axes', labelsize=LABEL_FS)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_FS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_FS)   # fontsize of the tick labels
plt.rc('legend', fontsize=LABEL_FS)  # legend fontsize
plt.rc('figure', titlesize=TITLE_FS) # fontsize of the figure title


# ==================================================================


class data_plotter:
    def __init__(self, var, band, data_handler, viewport, global_scale):
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
        global_scale : bool
            whether or not to scale the y-axis of all SAI variable figures to a common range
        '''

        self.var  = var
        self.band = band
        self.data_handler = data_handler
        self.viewport = viewport
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

        # ---- get data
        self.time = time2day(self.data_handler.data_avg_bands[self.band]['time'])
        self.var_data     = self.data_handler.data_avg_bands[self.band][self.var]
        if(self.data_handler.data_has_std):
            self.var_std_data = self.data_handler.data_std_avg_bands[self.band][self.var]
        
        if(self.data_handler.anom_base == 'Mean Climate'):
            self.anom_base_data = -(self.data_handler.anom_bands[self.band][self.var] - self.var_data)
        else:
            self.anom_base_data     = self.data_handler.anom_base_avg_bands[self.band][self.var]
            if(self.data_handler.anom_base_has_std):
                self.anom_base_std_data = self.data_handler.anom_base_std_avg_bands[self.band][self.var]

        # ---- get data limits
        self.datalim = [self.data_handler.avg_bands_min[var], self.data_handler.avg_bands_max[var]]
        self.datadel = abs(self.datalim[0] - self.datalim[1])

        # ---- get band limits
        bbounds = self.data_handler.band_bounds
        self.band_bounds = sorted((bbounds.tolist()[::-1] + (-bbounds[1:]).tolist())[band])

    # ==================================================================

    def plot_data(self):
        '''
        Plots the requested data and formats the figure
        '''
        
        # ---- create axis
        self.ax = self.viewport.figure.add_subplot(111)
        
        # ---- plot data
        if(self.data_handler.anom_base_has_std):
            self.ax.fill_between(self.time, self.anom_base_data - self.anom_base_std_data, 
                                            self.anom_base_data + self.anom_base_std_data, 
                                            color='k', alpha=0.2)
        self.ax.plot(self.time, self.anom_base_data, '-k', alpha=0.85, 
                     lw=1.5, label='{}'.format(self.data_handler.anom_base))
        
        if(self.data_handler.data_has_std):
            self.ax.fill_between(self.time, self.var_data - self.var_std_data, 
                                            self.var_data + self.var_std_data, 
                                            color=VAR_COLOR[self.var], alpha=0.2)
        self.ax.plot(self.time, self.var_data, '-', color=VAR_COLOR[self.var], alpha=0.85,
                     lw=1.5, label=self.data_handler.dataset)
        
        # ---- format figure
        self.ax.set_xlim(min(self.time), max(self.time))
        if(self.var[0] != 'T' and self.global_scale):
            self.ax.set_ylim(self.datalim[0] - self.datadel*0.05, 
                             self.datalim[1] + self.datadel*0.05)
        self.ax.grid()
        self.ax.legend()

        self.ax.set_xlabel('time [days]')
        if(VAR_UNIT[self.var] is not None): 
            self.ax.set_ylabel('{} [{}]'.format(VAR_NAME[self.var], VAR_UNIT[self.var]))
        else:
            self.ax.set_ylabel('{}'.format(VAR_NAME[self.var]))

        self.ax.set_title('Average {} for latitude band {} degrees\nData release: {}, Dataset: {}'.format( 
                                   self.var, self.band_bounds, self.data_handler.data_release, 
                                   self.data_handler.dataset), fontsize = TITLE_FS)
        
        # ---- done; update graphics
        self.viewport.canvas.draw()


# ==================================================================


