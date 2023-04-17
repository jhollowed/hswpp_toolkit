# Joe Hollowed
# University of Michigan 2023
#
# Providing a set of utility functions for use elsewhere in the package

import cftime
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------

# pdb breakpoints don't always work with Qt; workaround from
# https://stackoverflow.com/questions/1736015/debugging-a-pyqt4-app
def pyqt_set_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt5.QtCore import pyqtRemoveInputHook
    import pdb
    import sys
    pyqtRemoveInputHook()
    # set up the debugger
    debugger = pdb.Pdb()
    debugger.reset()
    # custom next to get outside of function scope
    debugger.do_next(None) # run the next command
    users_frame = sys._getframe().f_back # frame where the user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)

# ---------------------------------------------------------------------

def raise_error(str):
    '''Print an error message in red'''
    raise RuntimeError('\033[91m{}\033[0m'.format(str))

# ---------------------------------------------------------------------

def time2day(time):
    '''
    Converts cftime times to number of days since first timestamp

    Parameters
    ----------
    time : array of type from cftime
        array of times, e.g. an xarray DataSet, DataArray, numpy.array...

    Returns
    -------
    Number of days since time 0, for all times in the input
    '''
    if(type(time) == xr.core.dataarray.DataArray):
        time = np.array(time)
    start = '{}-{}-{}'.format(time[0].year, time[0].month, time[0].day)
    res = cftime.date2num(time, 'days since {}'.format(start))
    
    if(len(res) == 1): res = res[0]
    return res

# ---------------------------------------------------------------------

def clear_layout(layout):
    '''
    Clear a Qt Widget of it's layout and all children
    '''
    if(layout is None): return
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
