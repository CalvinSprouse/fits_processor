import argparse
import logging
import os
import time
import warnings
from pathlib import Path

import ccdproc
from astropy import units

from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyWarning
from rich.logging import RichHandler

# suppresses the fits fixed warning (annoying)
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# configure loggers one for terminal one for file errors
log = logging.Logger(name="DataFixerLog")
formatter = logging.Formatter("%(name)s|%(asctime)s|[%(levelname)s]|:%(message)s")
log.setLevel(logging.INFO)

stream_handler = RichHandler()
stream_handler.setLevel(logging.DEBUG) # change this to change terminal readout
file_handler = logging.FileHandler(filename="debug.log", delay=True)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR) # change this to change what is logged to file

log.addHandler(stream_handler)
log.addHandler(file_handler)


def add_filter(dir: str):
    filter_dict = {"O_III": ["O3", "OIII"],
                   "H_Beta": ["HB", "HBeta"],
                   "H_Alpha": ["HA", "HAlpha"],
                   "S_II": ["S2", "SII"],
                   "Bessel_V": ["BV", "BesselV"],
                   "Bessel_R": ["BR", "BesselR"],
                   "Bessel_U": ["BU", "BesselU"],
                   }
    for file in ccdproc.ImageFileCollection(dir).files_filtered(include_path=True):
        # get fits header
        fits = CCDData.read(file)

        # extract filter from filename
        for filter_name, filter_list in filter_dict.items():
            if any(["_" + f + "_" in file for f in filter_list]):
                log.info(f"File {file} identified as having filter {filter_name}.")
                fits.meta["FILTER"] = filter_name
                fits.write(file, overwrite=True)
                break


def add_bunit(dir: str):
    for file in ccdproc.ImageFileCollection(dir).files_filtered(include_path=True):
        # get fits header
        fits = CCDData.read(file, unit="adu")

        # check for fits header and add if not there
        if not fits.meta.get("BUNIT"):
            fits.meta["BUNIT"] = 'adu'

            # save
            fits.write(file, overwrite=True)
            log.info(f"BUNIT='adu' added to {file}.")

if __name__ == "__main__":
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("DataDir", type=str,
                        help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("-u", "--unit", action="store_true",
                        help="Ensure all data has the BUNIT='adu' fits data entry.")
    parser.add_argument("-f", "--filter", action="store_true",
                        help="Add FILTER= to fits header based on filename")
    args = parser.parse_args().__dict__

    # ensure theres is raw data to sort
    if not os.path.isdir(args["DataDir"]):
        log.error(f"User specified path {args['DataDir']} does not exist.")
        exit()

    if args["unit"]:
        add_bunit(args["DataDir"])

    if args["filter"]:
        add_filter(args["DataDir"])