# Joe Hollowed
# University of Michigan 2023
# 
# This class provides methods for reading and reducing zonally averaged datasets, and communicating results to callers from the GUI

import os
import sys
import time
import glob
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
CF_VAR_LIST = ['T025', 'T050', 'T1000']
MASTER_VAR_LIST = ['SO2', 'SULFATE', 'AOD', 'T025', 'T050', 'T1000']
MASTER_VAR_FMT = ['{:.2e}', '{:.2e}', '{:.2f}', '{:.1f}', '{:.1f}', '{:.1f}']

DATA_DIR = '{}/data'.format(pathlib.Path(__file__).parent.resolve())
PROCESSED_DIR = '{}/processed'.format(DATA_DIR)
DATA_TEMPLATE = {'011423':'HSW_SAI_ne16pg2_L72_1200day_180delay_{ENS}__MSO2_{MASSMAG}.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc',
                 '030123':'HSW_SAI_ne16pg2_L72_1200day_90delay__{ENS}_mass{MASSMAG}.eam.h2.0001-01-01-00000.regrid.91x180_aave.zonalMean.concat.nc'}
MEAN_CLIMATE = {'011423':'HSW_ne16pg2_L72_meanClimate.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc', 
                '030123':'HSW_ne16pg2_L72_meanClimate.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc'}
COUNTER_FACTUAL_TEMPLATE = {'011423':'HSW_SAI_ne16pg2_L72_1200day_180delay_{ENS}__cf.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc', 
                            '030123':''}


# ---- for updating progress bar
PYFILE = os.path.abspath(__file__)
p_increment = 100 / (sum(1 for line in open(os.path.abspath(__file__)) if 'incr_pbar(self.pbar)' in line) - 1)
def incr_pbar(pbar): 
    pbar.setProperty("value", pbar.value() + p_increment)
    QApplication.processEvents()


# ==================================================================


class data_handler:
    def __init__(self, data_release, dataset, mass_mag, trac_pres, 
                       anom_base, anom_def, anom_n, band_bounds, pbar, pbutton, overwrite=False):
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
            Anomaly definition, either ''
        anom_n : float
            Multiplicative factor to use for the anomaly definition
        band_bounds : list of float arrays
            List of bounds for four latitude bands, each given as a float np.array of length-2
        pbar : QProgressBar object
            handle to the progress bar object
        pbutton : QPushButton
            handle to the results refresh button, so its text can be updated
        overwrite : boolean
            whether or not to allow class to force overwrite processed data files (ensemble means, 
            band averages, etc.) even if files from previous runs exist. Will slow preformance,
            can potentially help clean data files if a preious run went wrong and/or encountered a bug.
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
        self.overwrite    = overwrite

        self.top_dir            = None
        self.data_file          = None
        self.data_std_file      = None
        self.data               = None
        self.data_std           = None
        self.data_has_std       = False
        self.anom_base_file     = None
        self.anom_base_data     = None
        self.anom_base_std_file = None
        self.anom_base_std_data = None
        self.anom_base_has_std  = False

        self.data_avg_band1  = None
        self.data_avg_band2N = None
        self.data_avg_band2S = None
        self.data_avg_band3N = None
        self.data_avg_band3S = None
        self.data_avg_band4N = None
        self.data_avg_band4S = None
        self.data_avg_bands  = None
        
        self.data_std_avg_band1  = None
        self.data_std_avg_band2N = None
        self.data_std_avg_band2S = None
        self.data_std_avg_band3N = None
        self.data_std_avg_band3S = None
        self.data_std_avg_band4N = None
        self.data_std_avg_band4S = None
        self.data_std_avg_bands  = None

        self.anom_base_avg_band1  = None
        self.anom_base_avg_band2N = None
        self.anom_base_avg_band2S = None
        self.anom_base_avg_band3N = None
        self.anom_base_avg_band3S = None
        self.anom_base_avg_band4N = None
        self.anom_base_avg_band4S = None
        self.anom_base_avg_bands  = None
        
        self.anom_base_std_avg_band1  = None
        self.anom_base_std_avg_band2N = None
        self.anom_base_std_avg_band2S = None
        self.anom_base_std_avg_band3N = None
        self.anom_base_std_avg_band3S = None
        self.anom_base_std_avg_band4N = None
        self.anom_base_std_avg_band4S = None
        self.anom_base_std_avg_bands  = None
        
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

    def load_data(self, force_zero_sai_cf=True):
        '''
        Load the data from file. The file is identified by specified run options at initialization

        Parameters
        ----------
        force_zero_sai_cf : bool
            whether or not to force tracer quantities in counterfactual runs to zero.
            This option is offered since there were some counterfactual runs which disables
            aerosol radiative feedback, but still advected passive tracer constituents. To
            avoid confusion, the quantities associated with these passive tracers are set to 
            zero/
        '''

        # ---- if ens_mean was requested, search processed dir, else search downloaded data 
        if(self.dataset == 'ens_mean'):
            self.top_dir = PROCESSED_DIR
            self.data_has_std = True
            if(self.anom_base == 'Counterfactual'): self.anom_base_has_std = True
        else:
            self.top_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        
        # ---- get dataset file by formatting the DATA_TEMPLATE string
        self.data_file = DATA_TEMPLATE[self.data_release]
        self.data_file = self.data_file.replace('{ENS}', self.dataset)
        self.data_file = self.data_file.replace('{MASSMAG}', self.mass_mag)
        self.data_std_file = self.data_file.replace('.nc', '.std.nc')

        # ---- open dataset if exists, else handle
        try:
            if(self.dataset == 'ens_mean'):
                if(self.overwrite): raise FileNotFoundError
                self.data_std = xr.load_dataset('{}/{}'.format(self.top_dir, self.data_std_file))
            self.data = xr.load_dataset('{}/{}'.format(self.top_dir, self.data_file))
        except FileNotFoundError:
            
            if(self.dataset == 'ens_mean'):
                self.compute_ens_mean()
                self.data     = xr.load_dataset('{}/{}'.format(self.top_dir, self.data_file))
                self.data_std = xr.load_dataset('{}/{}'.format(self.top_dir, self.data_std_file))
            
            else: raise_error('file {} not found; '\
                              'maybe delete downloaded data files and rerun'.format(self.data_file))
        incr_pbar(self.pbar)
                
        # ---- get selected data, coords, take data at requested pressure level for 3d fields
        self.data = self.data[MASTER_VAR_LIST]
        self.data = self.data.sel({'lev':self.trac_pres}, method='nearest')
        if(self.data_std is not None):
            # likewise for std
            self.data_std = self.data_std[MASTER_VAR_LIST]
            self.data_std = self.data_std.sel({'lev':self.trac_pres}, method='nearest')
        else:
            # if std has not beed defined, there is none; set to zero
            self.data_std = xr.zeros_like(self.data)

        print('---- data read from {}'.format(self.data_file))
        incr_pbar(self.pbar)
        
        # ---- anomaly base dataset
        self.top_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        if(self.anom_base == 'Mean Climate'):
            self.anom_base_file = MEAN_CLIMATE[self.data_release]
        elif(self.anom_base == 'Counterfactual'):
            self.anom_base_file = COUNTER_FACTUAL_TEMPLATE[self.data_release]
            self.anom_base_file = self.anom_base_file.replace('{ENS}', self.dataset)
            self.anom_base_std_file = self.data_file.replace('.nc', '.std.nc')
        
        # ---- open dataset if exists, else handle
        try:
            if(self.dataset == 'ens_mean' and self.anom_base == 'Counterfactual'):
                if(self.overwrite): raise FileNotFoundError
                self.anom_base_data_std = xr.load_dataset('{}/{}'.format(self.top_dir, self.anom_base_file))
            self.anom_base_data = xr.load_dataset('{}/{}'.format(self.top_dir, self.anom_base_file))
        except FileNotFoundError: 
            if(self.dataset == 'ens_mean' and self.anom_base == 'Counterfactual'):
                self.compute_cf_ens_mean()
                self.anom_base_data     = xr.load_dataset('{}/{}'.format(
                                          self.top_dir, self.anom_base_data_file))
                self.anom_base_data_std = xr.load_dataset('{}/{}'.format(
                                          self.top_dir, self.anom_base_data_std_file))
            else: raise_error('file {} not found; '\
                              'maybe delete downloaded data files and rerun'.format(self.anom_base_file))
        incr_pbar(self.pbar)
        
        # ---- counterfactuals and mean climate files may not have these SAI variables; if not, add zero-fields
        if ('SO2' not in self.anom_base_data.variables or force_zero_sai_cf):
            zeros = xr.zeros_like(self.anom_base_data['T025']).assign_attrs(self.data['SO2'].attrs)
            self.anom_base_data['SO2'] = zeros
        if ('SULFATE' not in self.anom_base_data.variables or force_zero_sai_cf):
            zeros = xr.zeros_like(self.anom_base_data['T025']).assign_attrs(self.data['SULFATE'].attrs)
            self.anom_base_data['SULFATE'] = zeros
        if ('AOD' not in self.anom_base_data.variables or force_zero_sai_cf):
            zeros = xr.zeros_like(self.anom_base_data['T025']).assign_attrs(self.data['AOD'].attrs)
            self.anom_base_data['AOD'] = zeros
    
        # ---- get selected data, coords, take data at requested pressure level for 3d fields
        self.anom_base_data = self.anom_base_data[MASTER_VAR_LIST]
        try:
            self.anom_base_data = self.anom_base_data.sel({'lev':self.trac_pres}, method='nearest')
        except KeyError:
            # if the above line doesn't work, the anom_base_data has no variables remaining with a 'lev'
            # coordinate. This is the case for e.g. the Mean Climate
            pass
        if(self.anom_base_std_data is not None):
            # likewise for std
            self.anom_base_data_std = self.data_std[MASTER_VAR_LIST]
            self.anom_base_data_std = self.data_std.sel({'lev':self.trac_pres}, method='nearest')
        else:
            # if std has not beed defined, there is none; set to zero
            self.anom_base_data_std = xr.zeros_like(self.anom_base_data)
        
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
        
        print('---- anomaly base data read from {}'.format(self.anom_base_file))
        incr_pbar(self.pbar)

         
    # ==================================================================
    

    def compute_ens_mean(self):
        '''
        Compute the ensemble mean and std of the selected data release
        '''

        # ---- get all ensemble members
        self.release_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        file_name_pattern = DATA_TEMPLATE[self.data_release]
        file_name_pattern = file_name_pattern.replace('{MASSMAG}', self.mass_mag)
        file_name_pattern = file_name_pattern.replace('{ENS}', 'ens[0-9][0-9]')
        ens_members_files = glob.glob('{}/{}'.format(self.release_dir, file_name_pattern))
        ens_members = [xr.load_dataset(d)[MASTER_VAR_LIST] for d in ens_members_files]
        N = len(ens_members)

        # ---- create xarray object for ens_mean and ens_std
        ens_mean = xr.zeros_like(ens_members[0])
        ens_std = xr.zeros_like(ens_members[0])
        
        # ---- loop through members, build mean and std
        print('---- Computing ensemble mean and std from {} members...'.format(N))
        for member in ens_members:
            ens_mean = ens_mean + member
        ens_mean = ens_mean / N
        print('    mean done')
        for member in ens_members:
            ens_std = ens_std + (member - ens_mean)**2
        ens_std = ens_std / (N-1)
        ens_std = np.sqrt(ens_std)
        print('    std done')

        # ---- done; close files and write out result
        for member in ens_members: member.close()
        print('    writing {}'.format(self.data_file))
        ens_mean.to_netcdf('{}/{}'.format(self.top_dir, self.data_file))
        print('    writing {}'.format(self.data_std_file))
        ens_std.to_netcdf('{}/{}'.format(self.top_dir, self.data_std_file))
        ens_mean.close()
        ens_std.close()
    
     
    # ==================================================================
    

    def compute_cf_ens_mean(self):
        '''
        Compute the ensemble mean and std of the selected data release
        '''
        
        # ---- get all ensemble members
        self.release_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        file_name_pattern = COUNTER_FACTUAL_TEMPLATE[self.data_release]
        file_name_pattern = file_name_pattern.replace('{ENS}', 'pert[0-9][0-9]')
        ens_members_files = glob.glob('{}/{}'.format(self.release_dir, file_name_pattern))
        ens_members = [xr.load_dataset(d)[CF_VAR_LIST] for d in ens_members_files]
        N = len(ens_members)

        # ---- create xarray object for ens_mean and ens_std
        ens_mean = xr.zeros_like(ens_members[0])
        ens_std = xr.zeros_like(ens_members[0])
        
        # ---- loop through members, build mean and std
        print('---- Computing ensemble mean and std from {} members...'.format(N))
        for member in ens_members:
            ens_mean = ens_mean + member
        ens_mean = ens_mean / N
        print('    mean done')
        for member in ens_members:
            ens_std = ensstd + (member - ens_mean)**2
        ens_std = ens_std / (N-1)
        ens_std = np.sqrt(ens_std)
        print('    std done')

        # ---- done; write out result
        for member in ens_members: member.close()
        print('    writing {}'.format(self.anom_base_data_file))
        ens_mean.to_netcdf('{}/{}'.format(self.top_dir, self.anom_base_file))
        print('    writing {}'.format(self.anom_base_std_file))
        ens_std.to_netcdf('{}/{}'.format(self.top_dir, self.anom_base_std_file))
        ens_mean.close()
        ens_std.close()

        
            
     # ==================================================================
    

    def average_lat_bands(self):
        '''
        Averages the data over the specified latitude bands, with cosine-latitude weighting. After
        averaging, the result per-band will be written to files at PROCESSED_DIR for reading on the 
        next run of this method with fixed bounds. The function will first attempt to read this file, 
        and only compute the meridional averages if these files are not found.
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
            if(self.overwrite): raise FileNotFoundError
            self.data_avg_band1 = xr.load_dataset(fname)
            print('read averaged data for band {} from {}'.format(band, fname.split('/')[-1]))
        except FileNotFoundError:
            self.data_avg_band1 = self.data.sel({'lat' : slice( *band)})
            self.data_avg_band1 = (self.data_avg_band1.weighted(dweights)).mean('lat')
            self.data_avg_band1.to_netcdf(fname)
            print('wrote averaged data for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        # --- std
        if(self.data_has_std):
            band = self.band_bounds[0]
            fname = '{}/{}_{}_Band{}.nc'.format(PROCESSED_DIR, self.data_release, 
                                                 self.data_std_file.split('.nc')[0], band)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.data_std_avg_band1 = xr.load_dataset(fname)
                print('read averaged data_std for band {} from {}'.format(band, fname.split('/')[-1]))
            except FileNotFoundError:
                self.data_std_avg_band1 = self.data_std.sel({'lat' : slice( *band)})
                self.data_std_avg_band1 = (self.data_std_avg_band1.weighted(dweights)).mean('lat')
                self.data_std_avg_band1.to_netcdf(fname)
                print('wrote averaged data_std for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- equatorial band - anomaly base
        fname = '{}/{}_{}_Band{}.nc'.format(PROCESSED_DIR, self.data_release, 
                                             self.anom_base_file.split('.nc')[0], band)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.anom_base_avg_band1 = xr.load_dataset(fname)
            print('read averaged anomaly base data for band {} from {}'.format(band, fname.split('/')[-1]))
        except FileNotFoundError:
            self.anom_base_avg_band1 = self.anom_base_data.sel({'lat' : slice( *band)})
            self.anom_base_avg_band1 = (self.anom_base_avg_band1.weighted(aweights)).mean('lat')
            self.anom_base_avg_band1.to_netcdf(fname)
            print('wrote averaged anomaly base data for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        # --- std
        if(self.anom_base_has_std):
            fname = '{}/{}_{}_Band{}.nc'.format(PROCESSED_DIR, self.data_release, 
                                                 self.anom_base_std_file.split('.nc')[0], band)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.anom_base_std_avg_band1 = xr.load_dataset(fname)
                print('read averaged anomaly base data for band {} from {}'.format(band, fname.split('/')[-1]))
            except FileNotFoundError:
                self.anom_base_std_avg_band1 = self.anom_base_std_data.sel({'lat' : slice( *band)})
                self.anom_base_std_avg_band1 = (self.anom_base_std_avg_band1.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band1.to_netcdf(fname)
                print('wrote averaged anomaly base data for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        
        
        #-------------------
        # ---- band 2 - data
        bandN = self.band_bounds[1]
        bandS = -self.band_bounds[1][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandS)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.data_avg_band2N = xr.load_dataset(fnameN)
            self.data_avg_band2S = xr.load_dataset(fnameS)
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
        # --- std
        if(self.data_has_std):
            bandN = self.band_bounds[1]
            bandS = -self.band_bounds[1][::-1]
            fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_std_file.split('.nc')[0], bandN)
            fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_std_file.split('.nc')[0], bandS)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.data_std_avg_band2N = xr.load_dataset(fnameN)
                self.data_std_avg_band2S = xr.load_dataset(fnameS)
                print('read averaged data_std for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
                print('read averaged data_std for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
            except FileNotFoundError:
                self.data_std_avg_band2N = self.data_std.sel({'lat' : slice(*bandN)})
                self.data_std_avg_band2S = self.data_std.sel({'lat' : slice(*bandS)})
                self.data_std_avg_band2N = (self.data_std_avg_band2N.weighted(dweights)).mean('lat')
                self.data_std_avg_band2S = (self.data_std_avg_band2S.weighted(dweights)).mean('lat')
                self.data_std_avg_band2N.to_netcdf(fnameN)
                self.data_std_avg_band2S.to_netcdf(fnameS)
                print('wrote averaged data_std for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
                print('wrote averaged data_std for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 2 - anomaly base
        bandN = self.band_bounds[1]
        bandS = -self.band_bounds[1][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandS)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.anom_base_avg_band2N = xr.load_dataset(fnameN)
            self.anom_base_avg_band2S = xr.load_dataset(fnameS)
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
        # --- std
        if(self.anom_base_has_std):
            bandN = self.band_bounds[1]
            bandS = -self.band_bounds[1][::-1]
            fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_std_file.split('.nc')[0], bandN)
            fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_std_file.split('.nc')[0], bandS)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.anom_base_std_avg_band2N = xr.load_dataset(fnameN)
                self.anom_base_std_avg_band2S = xr.load_dataset(fnameS)
                print('read averaged anom_base_std for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
                print('read averaged anom_base_std for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
            except FileNotFoundError:
                self.anom_base_std_avg_band2N = self.anom_base_std_data.sel({'lat' : slice(*bandN)})
                self.anom_base_std_avg_band2S = self.anom_base_std_data.sel({'lat' : slice(*bandS)})
                self.anom_base_std_avg_band2N = (self.anom_base_std_avg_band2N.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band2S = (self.anom_base_std_avg_band2S.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band2N.to_netcdf(fnameN)
                self.anom_base_std_avg_band2S.to_netcdf(fnameS)
                print('wrote averaged anom_base_std for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
                print('wrote averaged anom_base_std for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        
        #-------------------
        # ---- band 3 - data
        bandN = self.band_bounds[2]
        bandS = -self.band_bounds[2][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandS)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.data_avg_band3N = xr.load_dataset(fnameN)
            self.data_avg_band3S = xr.load_dataset(fnameS)
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
        # --- std
        if(self.data_has_std):
            bandN = self.band_bounds[2]
            bandS = -self.band_bounds[2][::-1]
            fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_std_file.split('.nc')[0], bandN)
            fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_std_file.split('.nc')[0], bandS)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.data_std_avg_band3N = xr.load_dataset(fnameN)
                self.data_std_avg_band3S = xr.load_dataset(fnameS)
                print('read averaged data_std for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
                print('read averaged data_std for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
            except FileNotFoundError:
                self.data_std_avg_band3N = self.data_std.sel({'lat' : slice(*bandN)})
                self.data_std_avg_band3S = self.data_std.sel({'lat' : slice(*bandS)})
                self.data_std_avg_band3N = (self.data_std_avg_band3N.weighted(dweights)).mean('lat')
                self.data_std_avg_band3S = (self.data_std_avg_band3S.weighted(dweights)).mean('lat')
                self.data_std_avg_band3N.to_netcdf(fnameN)
                self.data_std_avg_band3S.to_netcdf(fnameS)
                print('wrote averaged data_std for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
                print('wrote averaged data_std for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 3 - anomaly base
        bandN = self.band_bounds[2]
        bandS = -self.band_bounds[2][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandS)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.anom_base_avg_band3N = xr.load_dataset(fnameN)
            self.anom_base_avg_band3S = xr.load_dataset(fnameS)
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
        # --- std
        if(self.anom_base_has_std):
            bandN = self.band_bounds[2]
            bandS = -self.band_bounds[2][::-1]
            fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_std_file.split('.nc')[0], bandN)
            fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_std_file.split('.nc')[0], bandS)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.anom_base_std_avg_band3N = xr.load_dataset(fnameN)
                self.anom_base_std_avg_band3S = xr.load_dataset(fnameS)
                print('read averaged anom_base_std for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
                print('read averaged anom_base_std for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
            except FileNotFoundError:
                self.anom_base_std_avg_band3N = self.anom_base_std_data.sel({'lat' : slice(*bandN)})
                self.anom_base_std_avg_band3S = self.anom_base_std_data.sel({'lat' : slice(*bandS)})
                self.anom_base_std_avg_band3N = (self.anom_base_std_avg_band3N.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band3S = (self.anom_base_std_avg_band3S.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band3N.to_netcdf(fnameN)
                self.anom_base_std_avg_band3S.to_netcdf(fnameS)
                print('wrote averaged anom_base_std for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
                print('wrote averaged anom_base_std for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        #-------------------
        # ---- band 4 - data
        bandN = self.band_bounds[3]
        bandS = -self.band_bounds[3][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_file.split('.nc')[0], bandS)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.data_avg_band4N = xr.load_dataset(fnameN)
            self.data_avg_band4S = xr.load_dataset(fnameS)
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
        # --- std
        if(self.data_has_std):
            bandN = self.band_bounds[3]
            bandS = -self.band_bounds[3][::-1]
            fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_std_file.split('.nc')[0], bandN)
            fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.data_std_file.split('.nc')[0], bandS)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.data_std_avg_band4N = xr.load_dataset(fnameN)
                self.data_std_avg_band4S = xr.load_dataset(fnameS)
                print('read averaged data_std for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
                print('read averaged data_std for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
            except FileNotFoundError:
                self.data_std_avg_band4N = self.data_std.sel({'lat' : slice(*bandN)})
                self.data_std_avg_band4S = self.data_std.sel({'lat' : slice(*bandS)})
                self.data_std_avg_band4N = (self.data_std_avg_band4N.weighted(dweights)).mean('lat')
                self.data_std_avg_band4S = (self.data_std_avg_band4S.weighted(dweights)).mean('lat')
                self.data_std_avg_band4N.to_netcdf(fnameN)
                self.data_std_avg_band4S.to_netcdf(fnameS)
                print('wrote averaged data_std for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
                print('wrote averaged data_std for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 4 - anom_base
        bandN = self.band_bounds[3]
        bandS = -self.band_bounds[3][::-1]
        fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandN)
        fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_file.split('.nc')[0], bandS)
        try:
            if(self.overwrite): raise FileNotFoundError
            self.anom_base_avg_band4N = xr.load_dataset(fnameN)
            self.anom_base_avg_band4S = xr.load_dataset(fnameS)
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
        # --- std
        if(self.anom_base_has_std):
            bandN = self.band_bounds[3]
            bandS = -self.band_bounds[3][::-1]
            fnameN = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_std_file.split('.nc')[0], bandN)
            fnameS = '{}/{}_Band{}.nc'.format(PROCESSED_DIR, self.anom_base_std_file.split('.nc')[0], bandS)
            try:
                if(self.overwrite): raise FileNotFoundError
                self.anom_base_std_avg_band4N = xr.load_dataset(fnameN)
                self.anom_base_std_avg_band4S = xr.load_dataset(fnameS)
                print('read averaged anom_base_std for band {} from {}'.format(bandN, fnameN.split('/')[-1]))
                print('read averaged anom_base_std for band {} from {}'.format(bandS, fnameS.split('/')[-1]))
            except FileNotFoundError:
                self.anom_base_std_avg_band4N = self.anom_base_std_data.sel({'lat' : slice(*bandN)})
                self.anom_base_std_avg_band4S = self.anom_base_std_data.sel({'lat' : slice(*bandS)})
                self.anom_base_std_avg_band4N = (self.anom_base_std_avg_band4N.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band4S = (self.anom_base_std_avg_band4S.weighted(aweights)).mean('lat')
                self.anom_base_std_avg_band4N.to_netcdf(fnameN)
                self.anom_base_std_avg_band4S.to_netcdf(fnameS)
                print('wrote averaged anom_base_std for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
                print('wrote averaged anom_base_std for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- populate band lists for access by the data_plotter
        self.data_avg_bands          = [self.data_avg_band4N, self.data_avg_band3N,
                                        self.data_avg_band2N, self.data_avg_band1,
                                        self.data_avg_band2S, self.data_avg_band3S,
                                        self.data_avg_band4S]
        self.data_std_avg_bands      = [self.data_std_avg_band4N, self.data_std_avg_band3N,
                                        self.data_std_avg_band2N, self.data_std_avg_band1,
                                        self.data_std_avg_band2S, self.data_std_avg_band3S,
                                        self.data_std_avg_band4S]
        self.anom_base_avg_bands     = [self.anom_base_avg_band4N, self.anom_base_avg_band3N,
                                        self.anom_base_avg_band2N, self.anom_base_avg_band1,
                                        self.anom_base_avg_band2S, self.anom_base_avg_band3S,
                                        self.anom_base_avg_band4S]
        self.anom_base_std_avg_bands = [self.anom_base_std_avg_band4N, self.anom_base_std_avg_band3N,
                                        self.anom_base_std_avg_band2N, self.anom_base_std_avg_band1,
                                        self.anom_base_std_avg_band2S, self.anom_base_std_avg_band3S,
                                        self.anom_base_std_avg_band4S]
        
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
            
            var = MASTER_VAR_LIST[i]
            
            # -------- equatorial band
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band1[var] / self.data_std_avg_band1[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band1[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0
                
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
            #mask = self.anom_band2N[var] > (self.anom_n * self.anom_base_avg_band2N[var])
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band2N[var] / self.data_std_avg_band2N[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band2N[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0
            
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band2N['time'])[idx]
            anom_max = np.max(self.data_avg_band2N[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(2, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band2S
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band2S[var] / self.data_std_avg_band2S[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band2S[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0
            
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band2S['time'])[idx]
            anom_max = np.max(self.data_avg_band2S[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(4, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band3N
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band3N[var] / self.data_std_avg_band3N[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band3N[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0
            
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band3N['time'])[idx]
            anom_max = np.max(self.data_avg_band3N[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(1, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band3S
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band3S[var] / self.data_std_avg_band3S[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band3S[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0
            
            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band3S['time'])[idx]
            anom_max = np.max(self.data_avg_band3S[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(5, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band4N
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band4N[var] / self.data_std_avg_band4N[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band4N[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0

            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band4N['time'])[idx]
            anom_max = np.max(self.data_avg_band4N[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(0, i, result_text)
            incr_pbar(self.pbar)
            
            # -------- band4S
            if(var[0] == 'T' and self.anom_def == 'std'):
                mask = np.abs(self.anom_band4S[var] / self.data_std_avg_band4S[var]) > self.anom_n
            elif(var[0] == 'T' and self.anom_def == 'K'):
                mask = np.abs(self.anom_band4S[var]) > self.anom_n
            else:
                mask = self.anom_band1[var] > 0

            idx = (mask == True).argmax(dim='time').values.item()
            anom_onset_day = time2day(self.anom_band4S['time'])[idx]
            anom_max = np.max(self.data_avg_band4S[var].isel({'time':slice(idx, int(1e9))})).values.item()
            result_text = QTableWidgetItem('Day {:.0f}\nMax: {}'.format(anom_onset_day, 
                                                                        MASTER_VAR_FMT[i].format(anom_max)))
            table.setItem(6, i, result_text)
            incr_pbar(self.pbar)
            
              
# ==================================================================
