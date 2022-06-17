from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
from astropy import units as u
from pathlib import Path
from rich.logging import RichHandler

import ccdproc
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings

# suppress fits fixed warning
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

# configure logger/outs to terminal but will create file if errors
log = logging.Logger(name="DataReducerLog")
formatter = logging.Formatter("%(name)s|%(asctime)s|[%(levelname)s]|:%(message)s")
log.setLevel(logging.DEBUG)

stream_handler = RichHandler()
stream_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename="debug.log", delay=True)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

log.addHandler(stream_handler)
log.addHandler(file_handler)

log.info("Logger Established")

# configure file directories
raw_data_path = Path("RawData")
processed_data_path = Path("ProcessedData")
log.debug("Raw:{0}\tProcessed:{1}".format(raw_data_path, processed_data_path))

os.makedirs(processed_data_path, exist_ok=True)

# load in raw data/must be done with a glob sort
raw_bias = ccdproc.ImageFileCollection(location=raw_data_path, glob_include="[Bb]ias*")
log.debug("Loaded biases: {0}".format(raw_bias.files))
raw_darks = ccdproc.ImageFileCollection(location=raw_data_path, glob_include="[Dd]ark*")
log.debug("Loaded darks: {0}".format(raw_darks.files))
raw_flats = ccdproc.ImageFileCollection(location=raw_data_path, glob_include="[Ff]lat*")
log.debug("Loaded flats: {0}".format(raw_flats.files))

# create a master bias
master_bias = ccdproc.combine(raw_bias.files_filtered(include_path=True), method="median", unit="adu")
master_bias.meta["COMBINED"] = True
master_bias.write(Path(processed_data_path, "MasterBias.fits"), overwrite=True)
# TODO: Add overwrite parameter to determine re-run behavior
log.info("Master bias saved to {0}".format(processed_data_path))

# create master darks
dark_times = set(raw_darks.summary['exptime'])
log.debug("Found {0} as exposure times for dark images".format(dark_times))
for time in dark_times:
    selected_darks = raw_darks.files_filtered(exptime=time, include_path=True)
    log.debug("Creating {0}s master dark using {1}".format(time, selected_darks))

    master_dark = ccdproc.combine(selected_darks, method="median", unit="adu")
    master_dark.meta["COMBINED"] = True
    master_dark.write(Path(processed_data_path, "MasterDark{0}s.fits".format(int(time))), overwrite=True)
    log.info("Saved MasterDark{0}s.fits".format(time))
    # TODO: Make a filename maker which handles millisecond exposure times

    log.info("Master {0}s dark saved to {1}".format(time, processed_data_path))

# create master flats
# TODO: Find flexible alternative to assuming string positions
# creates a dictionary of filter to image list
flat_dict = {}
for file in raw_flats.files:
    split_file = file.split("_")
    im_filter = split_file[1].upper()

    if im_filter in flat_dict:
        flat_dict[im_filter].append(os.path.join(raw_data_path, file))
    else:
        flat_dict[im_filter] = [os.path.join(raw_data_path, file)]

# load all images of a given filter to create master flats
master_bias = CCDData.read(os.path.join(processed_data_path, "MasterBias.fits"), unit="adu")

for flat_filter, im_list in flat_dict.items():
    log.info("Loaded {0} filter with images: {1}".format(flat_filter, im_list))

    # first do standard median combination of flats
    master_flat = ccdproc.combine(im_list, method="median", unit="adu")
    log.debug("Combined flats for {0}".format(flat_filter))

    master_flat = ccdproc.subtract_bias(master_flat, master_bias)
    log.debug("Subtracted bias for {0}".format(flat_filter))

    # find same exposure darks by assuming all flats taken with the same exposure time
    # TODO: Handle the no darks error
    # TODO: Handle non-integer exposure times
    example_flat = CCDData.read(im_list[0], unit="adu")
    exposure = int(example_flat.meta["exptime"])
    log.debug("Determined exposure time to be {0} for {1}".format(exposure, flat_filter))

    # determined the equivalent dark through file name recreation
    # TODO: Do this better
    equivalent_dark = os.path.join(processed_data_path, "MasterDark{0}s.fits".format(exposure))
    log.debug("Determined equivalent dark to be {0} for {1}".format(equivalent_dark, flat_filter))

    master_flat = ccdproc.subtract_dark(master_flat, CCDData.read(equivalent_dark, unit="adu"),
                                        exposure_time=exposure, exposure_unit=u.second)
    log.debug("Subtracted darks for {0}".format(flat_filter))

    master_flat.write(os.path.join(processed_data_path, "MasterFlat{0}.fits".format(flat_filter)), overwrite=True)
    log.info("Saved MasterFlat{0}.fits".format(flat_filter))
