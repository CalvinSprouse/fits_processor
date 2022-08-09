import argparse
import ccdproc
import logging
import os
import time
import warnings
from pathlib import Path

from astropy import units
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

# suppress the fits fixed warning (astropy)
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# configure file log handling
log = logging.getLogger(__name__)

# create file handler
fhandler = logging.FileHandler(f".log")
fhandler.setLevel(logging.WARNING)
fhandler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# add file handler
log.addHandler(fhandler)
log.setLevel(logging.DEBUG)

# functions
def safe_write_ccddata(filename: Path, fits: CCDData, overwrite: bool = False):
    """ Save ccddata to a file location and prevent overwriting unless desired """
    pass


def create_master_bias(data_dir: Path, master_filename: Path, method: str = "median") -> Path:
    """ Create and save a master bias and return a path to the saved file """
    pass


def create_master_darks(data_dir: Path, master_save_dir: Path, method: str = "median") -> dict:
    """ Create and save master darks for all times in the data dir and return a dict of paths """
    pass


def create_master_flats(data_dir: Path, master_save_dir: Path, method: str = "median") -> dict:
    """ Create and save master flats for all filters found in the data dir using master files from master_save_dir """
    pass


def reduce_lights(data_dir: Path, reduced_data_dir: Path, object_sort: bool = False, filter_sort: bool = False, method: str = "median"):
    """ Reduce all lights found in the data dir and move the reduced files to the reduced data dir, optional sorting methods"""
    pass


if __name__ == "__main__":
    """ Run data reducer to.. reduce the data """
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("RawDir", type=str,
                        help="Path to the folder containing your raw fits files. (Can be relative)")

    args = parser.parse_args().__dict__

