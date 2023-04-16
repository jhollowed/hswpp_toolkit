# Joe Hollowed
# University of Michigan 2023
# 
# This class provides methods for reading and reducing zonally averaged datasets, and communicating results to callers from the GUI

import os
import time
import pathlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QTableWidgetItem

from util import time2day
from util import raise_error
from util import pyqt_set_trace as set_trace

from data_downloader import download_data


# ==================================================================


# ---- global vars
MASTER_VAR_LIST = ['SO2', 'SULFATE', 'AOD', 'T025', 'T050', 'T1000']
MASTER_COORD_LIST = ['lat', 'lev', 'time']
MASTER_VAR_FMT = ['{:.2e}', '{:.2e}', '{:.2f}', '{:.1f}', '{:.1f}', '{:.1f}']

DATA_DIR = '{}/data'.format(pathlib.Path(__file__).parent.resolve())
PROCESSED_DIR = '{}/processed'.format(DATA_DIR)
DATA_TEMPLATE = {'011423':'HSW_SAI_ne16pg2_L72_1200day_180delay_{ENS}.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc', '030123':''}
MEAN_CLIMATE = {'011423':'HSW_ne16pg2_L72_meanClimate.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc', '030123':''}
COUNTER_FACTUAL = {'011423':'', '030123':''}


# ---- for updating progress bar
PYFILE = os.path.abspath(__file__)
p_increment = 100 / (sum(1 for line in open(os.path.abspath(__file__)) if 'incr_pbar(self.pbar)' in line) - 1)
def incr_pbar(pbar): 
    pbar.setProperty("value", pbar.value() + p_increment)
    QApplication.processEvents()


# ==================================================================


class data_handler:
    def __init__(self, data_release, dataset, mass_mag, trac_pres, 
                       anom_base, anom_def, anom_n, band_bounds, pbar, pbutton):
        '''
        This object identifies data files corresponding to the options selected from the GUI, and
        offers methods for initiating computations on the data, exporting data to csv, and rendering plots

        Parameters
        ----------
        data_release : str
            The name of the data release to look for the requested datasets
        dataset : str
            Name of the dataset to use in the computations
        mass_mag : str
            SO2 injected mass multiplier
        trac_pres : float
            pressure at which to take horizontal tracer field (at nearest model level)
        anom_base : str
            Anomaly base dataset name
        anom_def : str
            Anomaly definition
        anom_n : float
            Multiplicative factor to use for the anomaly definition
        band_bounds : list of float arrays
            List of bounds for four latitude bands, each given as a float np.array of length-2
        pbar : QProgressBar object
            handle to the progress bar object
        pbutton : QPushButton
            handle to the results refresh button, so its text can be updated
        '''

        print('Constructing data handler with options:\n'\
              '    data release: {}\n'\
              '    dataset: {}\n'\
              '    pressure level: {}\n'
              '    SO2 mass magnitude: {}\n'\
              '    anomaly base: {}\n'\
              '    anomaly definiiton: {} x {}\n'\
              '    latitude bands: {}'.format(data_release, dataset, trac_pres, mass_mag, 
                                              anom_base, anom_n, anom_def, band_bounds))
        self.data_release = data_release
        self.dataset      = dataset
        self.trac_pres    = trac_pres
        self.mass_mag     = mass_mag
        self.anom_base    = anom_base
        self.anom_n       = anom_n
        self.anom_def     = anom_def
        self.band_bounds  = band_bounds
        self.pbar         = pbar

        self.release_dir    = None
        self.data_file      = None
        self.coords         = None
        self.data           = None
        self.anom_base_file = None
        self.anom_base_data = None

        self.data_avg_band1  = None
        self.data_avg_band2N = None
        self.data_avg_band2S = None
        self.data_avg_band3N = None
        self.data_avg_band3S = None
        self.data_avg_band4N = None
        self.data_avg_band4S = None
        self.data_avg_bands  = None

        self.anom_base_avg_band1  = None
        self.anom_base_avg_band2N = None
        self.anom_base_avg_band2S = None
        self.anom_base_avg_band3N = None
        self.anom_base_avg_band3S = None
        self.anom_base_avg_band4N = None
        self.anom_base_avg_band4S = None
        self.anom_base_avg_bands  = None
        
        self.anom_band1  = None
        self.anom_band2N = None
        self.anom_band2S = None
        self.anom_band3N = None
        self.anom_band3S = None
        self.anom_band4N = None
        self.anom_band4S = None
        self.anom_bands  = None

        self.avg_bands_max = None

        print('---- data_handler object initialized')
    
    # ==================================================================

    def load_data(self):
        '''
        Load the data from file. The file is identified by specified run options at initialization
        '''
        
        # --- open dataset
        self.release_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        self.data_file = DATA_TEMPLATE[self.data_release].replace('{ENS}', self.dataset)
        self.data = xr.open_dataset('{}/{}'.format(self.release_dir, self.data_file))
        
        self.coords = self.data[MASTER_COORD_LIST]
        self.data = self.data[MASTER_VAR_LIST]

        # take data at requested pressure level for 3d (tracer) fields
        self.data = self.data.sel({'lev':self.trac_pres}, method='nearest')

        print('---- data read from {}'.format(self.data_file))
        incr_pbar(self.pbar)
        
        # --- anomaly base dataset
        self.release_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        if(self.anom_base == 'Mean Climate'):
            self.anom_base_file = MEAN_CLIMATE[self.data_release]
        elif(self.anom_base == 'Counterfactual'):
            self.anom_base_file = COUNTER_FACTUAL[self.data_release]
        self.anom_base_data = xr.open_dataset('{}/{}'.format(self.release_dir, self.anom_base_file))
        
        # ---- verify data shapes
        if(self.anom_base == 'Counterfactual'):
            if(self.anom_base_data.dims != self.data.dims):
                raise_error('Counterfactual and chosen datasets do not have a matching time dimension!')
        if(self.anom_base == 'Mean Climate'):
            if('time' in self.anom_base_data.dims.keys()):
                if(len(self.anom_base_data['time']) > 1):
                    raise_error('Mean Climate data has len(time) > 1!')
                else:
                    self.anom_base_data = self.anom_base_data.isel(time=0)
      
        # ---- counterfactuals and mean climate files may not have these SAI variables; if not, add zero-fields 
        if 'SO2' not in self.anom_base_data.variables:
            zeros = xr.zeros_like(self.anom_base_data['T025']).assign_attrs(self.data['SO2'].attrs)
            self.anom_base_data['SO2'] = zeros
        if 'SULFATE' not in self.anom_base_data.variables:
            zeros = xr.zeros_like(self.anom_base_data['T025']).assign_attrs(self.data['SULFATE'].attrs)
            self.anom_base_data['SULFATE'] = zeros
        if 'AOD' not in self.anom_base_data.variables:
            zeros = xr.zeros_like(self.anom_base_data['T025']).assign_attrs(self.data['AOD'].attrs)
            self.anom_base_data['AOD'] = zeros
    
        self.anom_base_coords = self.anom_base_data[MASTER_COORD_LIST]
        self.anom_base_data = self.anom_base_data[MASTER_VAR_LIST]
        
        print('---- anomaly base data read from {}'.format(self.anom_base_file))
        incr_pbar(self.pbar)

                
         
    # ==================================================================
    
    def average_lat_bands(self, overwrite=False):
        '''
        Averages the data over the specified latitude bands, with cosine-latitude weighting. After
        averaging, the result per-band will be written to files at PROCESSED_DIR for reading on the 
        next run of this method with fixed bounds. The function will first attempt to read this file, 
        and only compute the meridional averages if these files are not found.

        Parameters
        ---------
        overwrite : bool
            whether or not to preform the averaging and overwrite processed data files, if they exist
        '''

        assert self.data is not None, 'Data not loaded! Call self.load_data() before self.average_lat_bands()'

        # --- weights are taken separately for each dataset, though they should be indentical
        dweights = np.cos(np.deg2rad(self.data['lat']))
        dweights.name = 'weights'
        incr_pbar(self.pbar)
        aweights = np.cos(np.deg2rad(self.anom_base_data['lat']))
        aweights.name = 'weights'
        incr_pbar(self.pbar)
        
        #----------------------------
        # ---- equatorial band - data
        band = self.band_bounds[0]
        fname = '{}/{}_{}_Band{}.nc'.format(PROCESSED_DIR, self.data_release, 
                                             self.data_file.split('.nc')[0], band)
        try:
            if(overwrite): raise FileNotFoundError
            self.data_avg_band1 = xr.open_dataset(fname)
            print('read averaged data for band {} from {}'.format(band, fname.split('/')[-1]))
        except FileNotFoundError:
            self.data_avg_band1 = self.data.sel({'lat' : slice( *band)})
            self.data_avg_band1 = (self.data_avg_band1.weighted(dweights)).mean('lat')
            self.data_avg_band1.to_netcdf(fname)
            print('wrote averaged data for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- equatorial band - anomaly base
        fname = '{}/{}_{}_Band{}.nc'.format(PROCESSED_DIR, self.data_release, 
                                             self.anom_base_file.split('.nc')[0], band)
        try:
            if(overwrite): raise FileNotFoundError
            self.anom_base_avg_band1 = xr.open_dataset(fname)
            print('read averaged anomaly base data for band {} from {}'.format(band, fname.split('/')[-1]))
        except FileNotFoundError:
            self.anom_base_avg_band1 = self.anom_base_data.sel({'lat' : slice( *band)})
            self.anom_base_avg_band1 = (self.anom_base_avg_band1.weighted(aweights)).mean('lat')
            self.anom_base_avg_band1.to_netcdf(fname)
            print('wrote averaged anomaly base data for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        
        
        #-------------------
        # ---- band 2 - data
        bandN = self.band_bounds[1]
        bandS = -self.band_bounds[1][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandS)
        try:
            if(overwrite): raise FileNotFoundError
            self.data_avg_band2N = xr.open_dataset(fnameN)
            self.data_avg_band2S = xr.open_dataset(fnameS)
            print('read averaged data for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
            print('read averaged data for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
        except FileNotFoundError:
            self.data_avg_band2N = self.data.sel({'lat' : slice(*bandN)})
            self.data_avg_band2S = self.data.sel({'lat' : slice(*bandS)})
            self.data_avg_band2N = (self.data_avg_band2N.weighted(dweights)).mean('lat')
            self.data_avg_band2S = (self.data_avg_band2S.weighted(dweights)).mean('lat')
            self.data_avg_band2N.to_netcdf(fnameN)
            self.data_avg_band2S.to_netcdf(fnameS)
            print('wrote averaged data for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged data for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 2 - anomaly base
        bandN = self.band_bounds[1]
        bandS = -self.band_bounds[1][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandS)
        try:
            if(overwrite): raise FileNotFoundError
            self.anom_base_avg_band2N = xr.open_dataset(fnameN)
            self.anom_base_avg_band2S = xr.open_dataset(fnameS)
            print('read averaged anom_base for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
            print('read averaged anom_base for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
        except FileNotFoundError:
            self.anom_base_avg_band2N = self.anom_base_data.sel({'lat' : slice(*bandN)})
            self.anom_base_avg_band2S = self.anom_base_data.sel({'lat' : slice(*bandS)})
            self.anom_base_avg_band2N = (self.anom_base_avg_band2N.weighted(aweights)).mean('lat')
            self.anom_base_avg_band2S = (self.anom_base_avg_band2S.weighted(aweights)).mean('lat')
            self.anom_base_avg_band2N.to_netcdf(fnameN)
            self.anom_base_avg_band2S.to_netcdf(fnameS)
            print('wrote averaged anom_base for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged anom_base for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        
        #-------------------
        # ---- band 3 - data
        bandN = self.band_bounds[2]
        bandS = -self.band_bounds[2][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandS)
        try:
            if(overwrite): raise FileNotFoundError
            self.data_avg_band3N = xr.open_dataset(fnameN)
            self.data_avg_band3S = xr.open_dataset(fnameS)
            print('read averaged data for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
            print('read averaged data for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
        except FileNotFoundError:
            self.data_avg_band3N = self.data.sel({'lat' : slice(*bandN)})
            self.data_avg_band3S = self.data.sel({'lat' : slice(*bandS)})
            self.data_avg_band3N = (self.data_avg_band3N.weighted(dweights)).mean('lat')
            self.data_avg_band3S = (self.data_avg_band3S.weighted(dweights)).mean('lat')
            self.data_avg_band3N.to_netcdf(fnameN)
            self.data_avg_band3S.to_netcdf(fnameS)
            print('wrote averaged data for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged data for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 3 - anomaly base
        bandN = self.band_bounds[2]
        bandS = -self.band_bounds[2][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandS)
        try:
            if(overwrite): raise FileNotFoundError
            self.anom_base_avg_band3N = xr.open_dataset(fnameN)
            self.anom_base_avg_band3S = xr.open_dataset(fnameS)
            print('read averaged anom_base for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
            print('read averaged anom_base for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
        except FileNotFoundError:
            self.anom_base_avg_band3N = self.anom_base_data.sel({'lat' : slice(*bandN)})
            self.anom_base_avg_band3S = self.anom_base_data.sel({'lat' : slice(*bandS)})
            self.anom_base_avg_band3N = (self.anom_base_avg_band3N.weighted(aweights)).mean('lat')
            self.anom_base_avg_band3S = (self.anom_base_avg_band3S.weighted(aweights)).mean('lat')
            self.anom_base_avg_band3N.to_netcdf(fnameN)
            self.anom_base_avg_band3S.to_netcdf(fnameS)
            print('wrote averaged anom_base for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged anom_base for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        #-------------------
        # ---- band 4 - data
        bandN = self.band_bounds[3]
        bandS = -self.band_bounds[3][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandS)
        try:
            if(overwrite): raise FileNotFoundError
            self.data_avg_band4N = xr.open_dataset(fnameN)
            self.data_avg_band4S = xr.open_dataset(fnameS)
            print('read averaged data for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
            print('read averaged data for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
        except FileNotFoundError:
            self.data_avg_band4N = self.data.sel({'lat' : slice(*bandN)})
            self.data_avg_band4S = self.data.sel({'lat' : slice(*bandS)})
            self.data_avg_band4N = (self.data_avg_band4N.weighted(dweights)).mean('lat')
            self.data_avg_band4S = (self.data_avg_band4S.weighted(dweights)).mean('lat')
            self.data_avg_band4N.to_netcdf(fnameN)
            self.data_avg_band4S.to_netcdf(fnameS)
            print('wrote averaged data for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged data for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 4 - anom_base
        bandN = self.band_bounds[3]
        bandS = -self.band_bounds[3][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandS)
        try:
            if(overwrite): raise FileNotFoundError
            self.anom_base_avg_band4N = xr.open_dataset(fnameN)
            self.anom_base_avg_band4S = xr.open_dataset(fnameS)
            print('read averaged anom_base for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
            print('read averaged anom_base for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
        except FileNotFoundError:
            self.anom_base_avg_band4N = self.anom_base_data.sel({'lat' : slice(*bandN)})
            self.anom_base_avg_band4S = self.anom_base_data.sel({'lat' : slice(*bandS)})
            self.anom_base_avg_band4N = (self.anom_base_avg_band4N.weighted(aweights)).mean('lat')
            self.anom_base_avg_band4S = (self.anom_base_avg_band4S.weighted(aweights)).mean('lat')
            self.anom_base_avg_band4N.to_netcdf(fnameN)
            self.anom_base_avg_band4S.to_netcdf(fnameS)
            print('wrote averaged anom_base for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged anom_base for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- populate band lists for access by the data_plotter
        self.data_avg_bands      = [self.data_avg_band4N, self.data_avg_band3N,
                                    self.data_avg_band2N, self.data_avg_band1,
                                    self.data_avg_band2S, self.data_avg_band3S,
                                    self.data_avg_band4S]
        self.anom_base_avg_bands = [self.anom_base_avg_band4N, self.anom_base_avg_band3N,
                                    self.anom_base_avg_band2N, self.anom_base_avg_band1,
                                    self.anom_base_avg_band2S, self.anom_base_avg_band3S,
                                    self.anom_base_avg_band4S]
        
        # ---- get min/max values for across bands per-var for access by the data plotter
        # -- max
        data_avg_bands_max = [dat.max() for dat in self.data_avg_bands]
        anom_base_avg_bands_max = [dat.max() for dat in self.anom_base_avg_bands]
        self.avg_bands_max = {}
        for var in MASTER_VAR_LIST:
            tmp = np.hstack([[dat[var] for dat in data_avg_bands_max],
                             [dat[var] for dat in anom_base_avg_bands_max]])
            self.avg_bands_max[var] = np.max(tmp)
        # -- min
        data_avg_bands_min = [dat.min() for dat in self.data_avg_bands]
        anom_base_avg_bands_min = [dat.min() for dat in self.anom_base_avg_bands]
        self.avg_bands_min = {}
        for var in MASTER_VAR_LIST:
            tmp = np.hstack([[dat[var] for dat in data_avg_bands_min],
                             [dat[var] for dat in anom_base_avg_bands_min]])
            self.avg_bands_min[var] = np.min(tmp)
        incr_pbar(self.pbar)
    
    
    # ==================================================================
    
    
    def compute_anomalies(self):
        '''
        Computes anomalies between the data and anomaly base
        '''
        
        assert self.data is not None,\
               'Data not loaded! Call self.load_data() before self.average_lat_bands()'
        assert self.data_avg_band1 is not None,\
               'Data not averaged! Call self.average_lat_bands() before self.average_lat_bands()'
        
        self.anom_band1  = self.data_avg_band1 - self.anom_base_avg_band1
        print('anomaly computed for band1')
        self.anom_band2N  = self.data_avg_band2N - self.anom_base_avg_band2N
        print('anomaly computed for band2N')
        self.anom_band2S  = self.data_avg_band2S - self.anom_base_avg_band2S
        print('anomaly computed for band2S')
        self.anom_band3N  = self.data_avg_band3N - self.anom_base_avg_band3N
        print('anomaly computed for band3N')
        self.anom_band3S  = self.data_avg_band3S - self.anom_base_avg_band3S
        print('anomaly computed for band3S')
        self.anom_band4N  = self.data_avg_band4N - self.anom_base_avg_band4N
        print('anomaly computed for band4N')
        self.anom_band4S  = self.data_avg_band4S - self.anom_base_avg_band4S
        print('anomaly computed for band4S')
        incr_pbar(self.pbar)
        
        # ---- populate band list for access by the data_plotter
        self.anom_bands          = [self.anom_band4N, self.anom_band3N,
                                    self.anom_band2N, self.anom_band1,
                                    self.anom_band2S, self.anom_band3S,
                                    self.anom_band4S]

    
    # ==================================================================

    def compute_benchmark_values(self, table):
        '''
        Computes the benchmark values which populate the results table panel of the GUI. These
        benchmarks are defined as:
        - day of anomaly onset
        - max. value post-anomaly onset
        for all variables in the MASTER_VAR_LIST

        Parameters
        ----------
        table : QTableWidget
            handle to the QTableWidget object in the GUI where benchmark results should be displayed
        '''
        
        anom_onset_day = 0
        anom_max = 0
        
        for i in range(len(MASTER_VAR_LIST)):
            
            # -------- equatorial band
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band1[var] > (self.anom_n * self.anom_base_avg_band1[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band1['time'])[idx]
            anom_max = np.max(self.data_avg_band1[var].isel({'time':slice(idx, int(1e9))})).values.item()

            # create a QTableWidgetItem with the variable as text
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            # set the item in the table at the current row and column
            table.setItem(3, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band2N
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band2N[var] > (self.anom_n * self.anom_base_avg_band2N[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band2N['time'])[idx]
            anom_max = np.max(self.data_avg_band2N[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(2, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band2S
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band2S[var] > (self.anom_n * self.anom_base_avg_band2S[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band2S['time'])[idx]
            anom_max = np.max(self.data_avg_band2S[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(4, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band3N
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band3N[var] > (self.anom_n * self.anom_base_avg_band3N[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band3N['time'])[idx]
            anom_max = np.max(self.data_avg_band3N[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(1, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band3S
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band3S[var] > (self.anom_n * self.anom_base_avg_band3S[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band3S['time'])[idx]
            anom_max = np.max(self.data_avg_band3S[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(5, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band4N
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band4N[var] > (self.anom_n * self.anom_base_avg_band4N[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band4N['time'])[idx]
            anom_max = np.max(self.data_avg_band4N[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(0, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band4S
            var = MASTER_VAR_LIST[i]
            mask = self.anom_band4S[var] > (self.anom_n * self.anom_base_avg_band4S[var])
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band4S['time'])[idx]
            anom_max = np.max(self.data_avg_band4S[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(6, i, result_text)
            incr_pbar(self.pbar)
            
            
        
    
    # ==================================================================

    def make_plots(self):
        return

              
# ==================================================================
