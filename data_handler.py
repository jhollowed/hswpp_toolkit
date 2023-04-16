# Joe Hollowed
# University of Michigan 2023
# 
# This class provides a graphical user interface for inspecting zonally-averaged variables for user-defined latitude bands in CLDERA HSW++ datasets

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pdb
import os
import time
from PyQt5.QtWidgets import QApplication
import requests
import pathlib
from bs4 import BeautifulSoup


# ==================================================================


# ---- global vars
MASTER_VAR_LIST = ['SO2', 'SULFATE', 'AOD', 'T025', 'T050', 'T1000']
MASTER_COORD_LIST = ['lat', 'lev', 'time']
DATA_DIR = '{}/data'.format(pathlib.Path(__file__).parent.resolve())
PROCESSED_DIR = './data/processed'
DATA_TEMPLATE = {'011423':'HSW_SAI_ne16pg2_L72_1200day_180delay_{ENS}.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc', '030123':''}
MEAN_CLIMATE = {'011423':'HSW_ne16pg2_L72_meanClimate.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc', '030123':''}
COUNTER_FACTUAL = {'011423':'', '030123':''}


# ---- for updating progress bar
PYFILE = os.path.abspath(__file__)
p_increment = 100 / (sum(1 for line in open(os.path.abspath(__file__)) if 'incr_pbar(self.pbar)' in line) - 1)
def incr_pbar(pbar): 
    pbar.setProperty("value", pbar.value() + p_increment)
    QApplication.processEvents()


# ---- for downloading data from Google Drive servers on first application run
def download_data():
  
    # point to Google Drive folders for each release
    releases = ['011423', '030123']
    dirs = ['{}/release_011423'.format(DATA_DIR), '{}/release_030123'.format(DATA_DIR)]
    urls = ['https://drive.google.com/drive/folders/1gik5ck-_kwjGx58Xmm2qSFsbqHbgeYPC?usp=share_link',
            'https://drive.google.com/drive/folders/1a0I0NSRk6YmkY7MmpLFhjC8SzKjCJ38n?usp=share_link']
    downloaded = 0
    print('\n---- Downloading data from server...')

    for i in range(len(releases)):
        release = releases[i]
        url = urls[i]
        data_dir = dirs[i]
        
        # open remote directory
        response = requests.get(url)
        if(response.status_code == 200):
            # get files in directory
            soup = BeautifulSoup(response.content, 'html.parser')
            file_links = soup.find_all('a', {'class':'drive-viewer-link'})
            print(file_links)from bs4 import BeautifulSoup
            files = response.json()['files']
            for fl in files:
                file_url = file['webContentLink']
                file_name = file['name']
                # skip if this file already exists locally
                if(os.path.isfile('{}/{}'.format(data_dir, file_name))):
                    continue
                # or download otherwise
                else:
                    print('downloading {}...'.format(file_name))
                    response = requests.get(file_url)
                    if(response.status_code == 200):
                        with open(file_name, 'wb') as f:
                            f.write(response.content)
                        downloaded  = downloaded + 1
                    else:
                        raise RuntimeError('Error downloading file {}: {}'.format(
                                                  file_name, response.status_code))
        else:
            raise RuntimeError('Error downloading release_{} data: {}'.format(release, response.status_code))
        
        if(downloaded > 0):
            print('--- release_{} data successfully downloaded'.format(release))
        else:
            print('--- release_{} data already exists'.format(release))
    
    

# ==================================================================


class data_handler:
    def __init__(self, data_release, dataset, mass_mag, anom_base, anom_def, anom_n, band_bounds, pbar):
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
        '''

        print('Constructing data handler with options:\n'\
              'data release: {}\n'\
              'dataset: {}\n'\
              'SO2 mass magnitude: {}\n'\
              'anomaly base: {}\n'\
              'anomaly definiiton: {} x {}\n'\
              'latitude bands: {}'.format(data_release, dataset, mass_mag, anom_base, 
                                          anom_n, anom_def, band_bounds))
        self.data_release = data_release
        self.dataset      = dataset
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
        
        self.anom_base_avg_band1  = None
        self.anom_base_avg_band2N = None
        self.anom_base_avg_band2S = None
        self.anom_base_avg_band3N = None
        self.anom_base_avg_band3S = None
        self.anom_base_avg_band4N = None
        self.anom_base_avg_band4S = None
        
        self.anom_band1  = None
        self.anom_band2N = None
        self.anom_band2S = None
        self.anom_band3N = None
        self.anom_band3S = None
        self.anom_band4N = None
        self.anom_band4S = None

        print('---- data_handler object initialized')
        
        # check that data exists, if not then download
        download_data()
            
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

        print('---- data read from {}'.format(self.data_file))
        incr_pbar(self.pbar)
        
        # --- anomaly base dataset
        self.release_dir = '{}/release_{}'.format(DATA_DIR, self.data_release)
        if(self.anom_base == 'Mean Climate'):
            self.anom_base_file = MEAN_CLIMATE[self.data_release]
        elif(self.anom_base == 'Counterfactual'):
            self.anom_base_file = COUNTER_FACTUAL[self.data_release]
        self.anom_base_data = xr.open_dataset('{}/{}'.format(self.release_dir, self.anom_base_file))
      
        # counter factuals and mean climate files may not have these SAI variables; if not, add zero-fields 
        if 'SO2' not in self.anom_base_data.variables:
            self.anom_base_data['SO2'] = xr.zeros_like(self.data['SO2'])
            self.anom_base_data['SO2'].encoding["_FillValue"] = 0.0
        if 'SULFATE' not in self.anom_base_data.variables:
            self.anom_base_data['SULFATE'] = xr.zeros_like(self.data['SULFATE'])
            self.anom_base_data['SULFATE'].encoding["_FillValue"] = 0.0
        if 'AOD' not in self.anom_base_data.variables:
            self.anom_base_data['AOD'] = xr.zeros_like(self.data['AOD'])
            self.anom_base_data['AOD'].encoding["_FillValue"] = 0.0

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

        weights = np.cos(np.deg2rad(self.data['lat']))
        weights.name = 'weights'
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
            self.data_avg_band1 = (self.data_avg_band1.weighted(weights)).mean('lat')
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
            self.anom_base_avg_band1 = (self.anom_base_avg_band1.weighted(weights)).mean('lat')
            self.anom_base_avg_band1.to_netcdf(fname)
            print('wrote averaged anomaly base data for band {} to {}'.format(band, fname.split('/')[-1]))
        incr_pbar(self.pbar)
        
        
        #-------------------
        # ---- band 2 - data
        bandN = self.band_bounds[1]
        bandS = -self.band_bounds[1]
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
            self.data_avg_band2N = (self.data_avg_band2N.weighted(weights)).mean('lat')
            self.data_avg_band2S = (self.data_avg_band2S.weighted(weights)).mean('lat')
            self.data_avg_band2N.to_netcdf(fnameN)
            self.data_avg_band2S.to_netcdf(fnameS)
            print('wrote averaged data for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged data for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 2 - anomaly base
        bandN = self.band_bounds[1]
        bandS = -self.band_bounds[1]
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
            self.anom_base_avg_band2N = (self.anom_base_avg_band2N.weighted(weights)).mean('lat')
            self.anom_base_avg_band2S = (self.anom_base_avg_band2S.weighted(weights)).mean('lat')
            self.anom_base_avg_band2N.to_netcdf(fnameN)
            self.anom_base_avg_band2S.to_netcdf(fnameS)
            print('wrote averaged anom_base for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged anom_base for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        
        #-------------------
        # ---- band 3 - data
        bandN = self.band_bounds[2]
        bandS = -self.band_bounds[2]
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
            self.data_avg_band3N = (self.data_avg_band3N.weighted(weights)).mean('lat')
            self.data_avg_band3S = (self.data_avg_band3S.weighted(weights)).mean('lat')
            self.data_avg_band3N.to_netcdf(fnameN)
            self.data_avg_band3S.to_netcdf(fnameS)
            print('wrote averaged data for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged data for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 3 - anomaly base
        bandN = self.band_bounds[2]
        bandS = -self.band_bounds[2]
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
            self.anom_base_avg_band3N = (self.anom_base_avg_band3N.weighted(weights)).mean('lat')
            self.anom_base_avg_band3S = (self.anom_base_avg_band3S.weighted(weights)).mean('lat')
            self.anom_base_avg_band3N.to_netcdf(fnameN)
            self.anom_base_avg_band3S.to_netcdf(fnameS)
            print('wrote averaged anom_base for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged anom_base for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        #-------------------
        # ---- band 4 - data
        bandN = self.band_bounds[3]
        bandS = -self.band_bounds[3]
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
            self.data_avg_band4N = (self.data_avg_band4N.weighted(weights)).mean('lat')
            self.data_avg_band4S = (self.data_avg_band4S.weighted(weights)).mean('lat')
            self.data_avg_band4N.to_netcdf(fnameN)
            self.data_avg_band4S.to_netcdf(fnameS)
            print('wrote averaged data for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged data for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
        incr_pbar(self.pbar)
        
        # ---- band 4 - anom_base
        bandN = self.band_bounds[3]
        bandS = -self.band_bounds[3]
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
            self.anom_base_avg_band4N = (self.anom_base_avg_band4N.weighted(weights)).mean('lat')
            self.anom_base_avg_band4S = (self.anom_base_avg_band4S.weighted(weights)).mean('lat')
            self.anom_base_avg_band4N.to_netcdf(fnameN)
            self.anom_base_avg_band4S.to_netcdf(fnameS)
            print('wrote averaged anom_base for band {} to {}'.format(bandN, fnameN.split('/')[-1]))
            print('wrote averaged anom_base for band {} to {}'.format(bandS, fnameS.split('/')[-1]))
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

    
    # ==================================================================

    def compute_benchmark_values(self):
        '''
        Computes the benchmark values which populate the results table panel of the GUI. These
        benchmarks are defined as:
        - day of anomaly onset
        - max. value post-anomaly onset
        for all variables in the MASTER_VAR_LIST
        '''

        
    
    # ==================================================================

    def make_plots(self):
        return

              
# ==================================================================
