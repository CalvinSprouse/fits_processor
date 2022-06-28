# __FILE NAME FORMAT__:
# {ImageType}_{Filter}_{ExposureTime}_{FileNumber}.fits

# __FILTER NAME FORMAT__:
# - Red = R
# - Blue = B
# - Green = G
# - Bessel R = BR
# - Bessel B = BB
# - Bessel G = BG
# - None = CLEAR
# - OIII = O3
# - SII = S2
# - Infrared = I
# - Bessel Infrared = BI
# - UV = UV
# - Bessel UV = BUV
# - H Beta = HB
# - H Alpha = HA

# imports
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
from astropy import units
from pathlib import Path
from rich.logging import RichHandler

import argparse
import ccdproc
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings

# suppresses the fits fixed warning (annoying)
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# configure loggers
# outputs to terminal but will save any errors to a log
log = logging.Logger(name="DataReducerLog")
formatter = logging.Formatter("%(name)s|%(asctime)s|[%(levelname)s]|:%(message)s")
log.setLevel(logging.DEBUG)

stream_handler = RichHandler()
stream_handler.setLevel(logging.DEBUG) # change this to change terminal readout
file_handler = logging.FileHandler(filename="debug.log", delay=True)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR) # change this to change what is logged to file

log.addHandler(stream_handler)
log.addHandler(file_handler)

# functions
def exp_time_to_str(exposure_time) -> str:
    """Convert an exposure time to string, used to normalize between formats"""
    if int(exposure_time) == exposure_time: 
        # checks for an integer exposure time greater than 1
        return str(int(exposure_time)) + "s"
    elif int(exposure_time*1000) == exposure_time*1000: 
        # checks for an in integer exposure time in milliseconds
        return str(int(exposure_time*1000)) + "ms"
    return str(exposure_time)

def create_master_bias(raw_images: ccdproc.ImageFileCollection, processed_data_path: Path, overwrite:bool = False) -> CCDData:
    """Create a master bias from raw_data_path and save it to processed_data_path"""
    # sort out biases from other images by IMTYPE
    raw_biases = raw_images.files_filtered(IMTYPE="Bias", include_path=True)
    # define the master file save path
    master_filename = os.path.join(processed_data_path, "MasterBias.fits") 

    # combine and save biases using a median combine TODO: Make configurable
    master_bias = ccdproc.combine(raw_biases, method="median", unit="adu")
    master_bias.write(master_filename, overwrite=overwrite)

    log.info("Created Master Bias from {0} images -> {1}".format(len(raw_biases), master_filename))
    return master_bias

def create_master_darks(raw_images: ccdproc.ImageFileCollection, processed_data_path: Path, overwrite:bool = False) -> dict:
    """Create master darks for all times from raw_data_path and save it to processed_data_path"""
    # find all unique exposure times
    dark_times = set([CCDData.read(t, unit="adu").meta.get("EXPTIME") for t in list(raw_images.files_filtered(IMTYPE="Dark", include_path=True))])
    master_darks = {}

    # each unique time gets a master dark which is saved to the dictionary
    for time in dark_times:
        # sort out darks of a specific exposure from other images by IMTYPE and EXPTIME
        selected_darks = raw_images.files_filtered(EXPTIME=time, IMTYPE="Dark", include_path=True)
        # define the master file save path
        save_time = exp_time_to_str(time)
        master_filename = os.path.join(processed_data_path, "MasterDark{0}.fits".format(save_time))

        # combine and save darks using a median combine
        master_dark = ccdproc.combine(selected_darks, method="median", unit="adu")
        master_dark.write(master_filename, overwrite=overwrite)

        # save the master dark to the dark dictionary
        master_darks[save_time] = CCDData.read(master_filename)

        log.info("Created {0} Master Dark from {1} images -> {2}".format(save_time, len(selected_darks), master_filename))
    
    return master_darks

def create_master_flats(raw_images: ccdproc.ImageFileCollection, raw_data_path: Path, processed_data_path: Path, 
                        master_bias: CCDData, master_darks_dict: dict, overwrite:bool = False) -> dict:
    """Create master flats for all filters from raw_data_path and save it to processed_data_path"""
    # list from the filter format above (WIP)
    raw_flats = {f:[] for f in ["R", "B", "G", "BR", "BG", "BB", "I", "H3", "S2", "HA", "HB", "CLEAR"]}

    # sort raw flats into the raw flats dict using above formatting (index 1 = filter)
    for flat in list(raw_images.files_filtered(IMTYPE="Flat")) + list(raw_images.files_filtered(IMTYPE="Sky")):
        # extract filter from the first index of the filename (this is important make sure files are formatted properly)
        flat_filter = flat.split("_")[1].upper()

        # save to the raw dictionary if that filter is recognized
        if flat_filter in raw_flats:
            raw_flats[flat_filter].append(os.path.join(raw_data_path, flat))

    # create master flats
    master_flats = {}

    for flat_filter, raw_list in raw_flats.items():
        # find all unique exposure times in the flat files
        flat_times = list(set(CCDData.read(os.path.join(t), unit="adu").meta.get("EXPTIME") for t in raw_list))

        # checks for uniformity in flat times and skips non-uniform flat collections
        # TODO: Handle this better (may not be neccesary but consider it)
        if len(flat_times) != 1:
            continue

        # since flat times are uniform after the above step flat_time and flat_time[0] are the same (len 1 array)
        flat_time = flat_times[0]
        # define the master save location
        master_filename = os.path.join(processed_data_path, "MasterFlat{0}.fits".format(flat_filter))

        # create the master flat with a median combine, bias subtraction, and dark subtraction from master files
        master_flat = ccdproc.combine(raw_list, method="median", unit="adu")
        master_flat = ccdproc.subtract_bias(master_flat, master_bias)
        master_flat = ccdproc.subtract_dark(master_flat, master_darks_dict[exp_time_to_str(flat_time)], 
                                            dark_exposure=flat_time * units.second, data_exposure=flat_time * units.second)
        master_flat.write(master_filename, overwrite=overwrite)

        # save the master flat to the flat dictionary
        master_flats[flat_filter] = CCDData.read(master_filename)

        log.info("Created {0} {1} Master Flat from {2} images -> {3}".format(flat_filter, exp_time_to_str(flat_time), len(raw_list), master_filename))
    
    return master_flats

def reduce_raw_lights(raw_images: ccdproc.ImageFileCollection, raw_data_path:Path, processed_data_path: Path, 
                      master_bias: CCDData, master_darks: dict, master_flats: dict):
    # get all unique objects
    objects_list = set([CCDData.read(l, unit="adu").meta.get("OBJECT") for l in raw_images.files_filtered(IMTYPE="Light", include_path=True)])

    for obj_observed in objects_list:
        # get the observed object for creating the save dir and sorting images
        obj_name = obj_observed.strip().replace(" ", "")
        obj_save_dir = os.path.join(processed_data_path, obj_name)
        os.makedirs(obj_save_dir, exist_ok=True)

        for image in raw_images.files_filtered(IMTYPE="Light", OBJECT=obj_observed):
            # define some variables for the raw image including its save path
            ccd_image = CCDData.read(os.path.join(raw_data_path, image), unit="adu")
            reduced_filename = os.path.join(obj_save_dir, "Reduced_" + image)
            exposure_time = ccd_image.meta.get("EXPTIME")
            exposure_string = exp_time_to_str(exposure_time)
            image_filter = image.split("_")[1].upper()

            # ensure the exposure time has been processed
            if exposure_string not in master_darks:
                continue

            # ensure the filter has been processed
            if image_filter not in master_flats:
                continue

            # create a reduced light by subtracting bias, darks, and doing a flat division
            reduced = ccdproc.subtract_bias(ccd_image, master_bias)
            reduced = ccdproc.subtract_dark(reduced, master_darks[exposure_string], 
                                            dark_exposure=exposure_time * units.second, data_exposure=exposure_time * units.second)
            reduced = ccdproc.flat_correct(reduced, master_flats[image_filter])
            reduced.write(reduced_filename, overwrite=True)

            log.info("Reduced {0} using {1} Filter {2} Exposure {3} Object -> {4}".format(image, image_filter, exposure_string, obj_name, reduced_filename))

def calibrate_and_reduce(raw_data_path: str, processed_data_path: str, overwrite:bool = False):
    """Create master bias, darks, and flats from raw_data and save them to processed data, returns all three master calibration images"""
    # define paths
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    # load all raw images found in target dir
    raw_images = ccdproc.ImageFileCollection(raw_data_path)
    log.info("Loaded {0} raw images".format(len(raw_images.files_filtered())))

    # ensure processed directory exists
    os.makedirs(processed_data_path, exist_ok=True)

    # create master images
    master_bias = create_master_bias(raw_images, processed_data_path, overwrite)
    master_darks = create_master_darks(raw_images, processed_data_path, overwrite)
    master_flats = create_master_flats(raw_images, raw_data_path, processed_data_path, master_bias, master_darks, overwrite)
    
    # use master images to reduce lights
    reduce_raw_lights(raw_images, raw_data_path, processed_data_path, master_bias, master_darks, master_flats)

# TODO: Fill out filter list

if __name__ == "__main__":
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("RawDataDir", type=str, help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("ProcessedDataDir", type=str, help="Path to the folder you want saved data to go. (WARNING: May overwrite existing data)")
    parser.add_argument("-o", "--overwrite", action="store_true", help="True if you want to overwrite existing files with the same name. (default: false)")
    args = parser.parse_args().__dict__

    # ensure theres is raw data to sort
    if not os.path.isdir(args["RawDataDir"]):
        log.warning("User specified path {0} does not exist".format(args["RawDataDir"]))
        exit()

    # TODO: Add more validation
    # run calibrate and reduce code
    calibrate_and_reduce(args["RawDataDir"], args["ProcessedDataDir"], args["overwrite"])