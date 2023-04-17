# Joe Hollowed
# University of Michigan 2023
#
# Functions for downloading reduced zonally-averaged HSW++ datasets from figshare

import os
import pathlib
import requests
from PyQt5.QtWidgets import QApplication

# ==================================================================


# ---- global vars
DATA_DIR = '{}/data'.format(pathlib.Path(__file__).parent.resolve())
INVENTORY ={'011423': {
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens02__MSO2_1.00X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184132',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens01__MSO2_1.00X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184138',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens03__MSO2_1.00X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184141',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens04__MSO2_1.00X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184150',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens05__MSO2_0.25X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184153',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens05__MSO2_0.50X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184156',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens05__MSO2_1.00X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184159',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens05__MSO2_1.50X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184162',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens05__MSO2_2.00X.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184165',
                'HSW_ne16pg2_L72_meanClimate.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184167',
                'HSW_SAI_ne16pg2_L72_1200day_180delay_ens05__cf.eam.h2.0001-01-01-00000.regrid.91x180_bilinear.zonalMean.nc':'https://figshare.com/ndownloader/files/40184168' },
            '030123': {}}

# ---- for updating progress bar
p_increment = 100/(len(INVENTORY['011423'].keys()) + len(INVENTORY['030123'].keys()))
def incr_pbar(pbar):
    pbar.setProperty("value", pbar.value() + p_increment)
    QApplication.processEvents()


# ==================================================================


def download_data(pbar):
    '''
    Function for downloading data from figshare servers on first application run

    Parameters
    ----------
    pbar : QProgressBar object
            handle to the progress bar object
    '''

    # loop over data releases
    releases = ['011423', '030123']
    dirs = ['{}/release_{}'.format(DATA_DIR, r) for r in releases]
    downloaded = 0
    print('\n---- Downloading data from server...')
    QApplication.processEvents()

    for i in range(len(releases)):
        release = releases[i]
        files = INVENTORY[release]
        data_dir = dirs[i]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        j = 0

        # download files
        for file_name in files.keys():
            file_path = '{}/{}'.format(data_dir, file_name)
            j = j+1

            # skip if this file already exists locally
            if(os.path.isfile(file_path)):
                print('skipping existing file {}/{}: {}...'.format(j, len(files), file_name))
                continue
            # or download otherwise
            else:
                print('downloading file {}/{}: {}...'.format(j, len(files), file_name))
                response = requests.get(files[file_name])
                if(response.status_code == 200):
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    downloaded  = downloaded + 1
                else:
                    raise RuntimeError('Error downloading file {}: {}'.format(
                                              file_name, response.status_code))
            # update progress bar
            incr_pbar(pbar)

        # done
        if(downloaded > 0):
            print('--- release_{} data successfully downloaded'.format(release))
        else:
            print('--- release_{} data already exists'.format(release))



# ==================================================================
