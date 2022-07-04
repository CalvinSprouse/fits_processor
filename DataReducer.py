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
log = logging.Logger(name="DataReducerLog")
formatter = logging.Formatter("%(name)s|%(asctime)s|[%(levelname)s]|:%(message)s")
log.setLevel(logging.INFO)

stream_handler = RichHandler()
stream_handler.setLevel(logging.DEBUG) # change this to change terminal readout
file_handler = logging.FileHandler(filename="debug.log", delay=True)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR) # change this to change what is logged to file

log.addHandler(stream_handler)
log.addHandler(file_handler)

# calibration functions
def safe_write_ccddata(fits: CCDData, filename: str, overwrite: bool = False) -> bool:
    """Writes .fits files and handles the file already exists error. Returns True/False if a file was written."""
    try:
        dir_name = Path(filename).parent.absolute()
        log.debug(f"Ensuring {dir_name} exists. overwrite={overwrite}")
        os.makedirs(dir_name, exist_ok=True)

        # data tag to alert the program this image was in fact processed by code (can be spoofed)
        fits.meta["PROGRAM_PROCESSED"] = True
        fits.write(filename, overwrite=overwrite)
        return True
    except OSError as e:
        log.warning(f"File {filename} already exists and overwrite={overwrite}. Skipping.")
    return False


def exposure_to_str(exposure_time) -> str:
    """Converts an exposure double to readable string"""
    if int(exposure_time) == exposure_time:
        # checks for an integer exposure time greater than 1
        return str(int(exposure_time)) + "s"
    elif int(exposure_time*1000) == exposure_time*1000:
        # checks for an in integer exposure time in milliseconds
        return str(int(exposure_time*1000)) + "ms"
    return str(exposure_time)


def get_master_calibration_file(master_save_dir: str, **search_args) -> CCDData:
    """Gets a master calibration file from the processed data dir based on search KWARGS"""
    try:
        master_fits = ccdproc.ImageFileCollection(master_save_dir).files_filtered(**search_args, PROGRAM_PROCESSED=True)
    except FileNotFoundError:
        log.error(f"The provided path {master_save_dir} does not exist.")
        exit()

    # for calibration files only 1 can be used for a given search term, function will not return more than 1
    if len(master_fits) == 1:
        return CCDData.read(os.path.join(master_save_dir, master_fits[0]))
    elif len(master_fits) == 0:
        log.error(f"No master calibration files found for the search args ({search_args})")
    else:
        log.error(f"Multiple master calibration files ({master_fits}) returned for the provided search args ({search_args})")
    exit()

def create_master_bias(raw_data_dir: str, processed_data_dir: str, data_method: str = "median", overwrite: bool = False):
    """Create a master bias from the raw data dir and save it to the processed data dir"""
    # get all biases from the raw data directory
    raw_biases = ccdproc.ImageFileCollection(raw_data_dir).files_filtered(IMTYPE="Bias", include_path=True)

    # define the master file save path
    master_filename = os.path.join(processed_data_dir, "MasterBias.fits")

    # combine and save using median combine
    master_bias = ccdproc.combine(raw_biases, method=data_method)
    if safe_write_ccddata(master_bias, master_filename, overwrite):
        log.info(f"Created Master Bias from {len(raw_biases)} images -> {master_filename}")


def create_master_darks(raw_data_dir: str, processed_data_dir: str, data_method: str = "median", overwrite: bool = False):
    """Create master darks for all times from the raw data dir and save it to the processed data dir"""
    # get a list of raw files
    raw_fits = ccdproc.ImageFileCollection(raw_data_dir)

    # get all unique dark exposure times
    dark_times = set([CCDData.read(fits).meta.get("EXPTIME") for fits in list(raw_fits.files_filtered(IMTYPE="Dark", include_path=True))])
    log.debug(f"Dark Times {dark_times}")

    # each unique time gets a master dark
    for time in dark_times:
        # sort darks of a specific exposure from other images by IMTYPE and EXPTIME
        selected_darks = raw_fits.files_filtered(EXPTIME=time, IMTYPE="Dark", include_path=True)

        # define the master file save path
        save_time = exposure_to_str(time)
        master_filename = os.path.join(processed_data_dir, f"MasterDark_{save_time}.fits")

        # combine and save
        master_dark = ccdproc.combine(selected_darks, method=data_method)
        if safe_write_ccddata(master_dark, master_filename, overwrite=overwrite):
            log.info(f"Created Master Dark from {len(selected_darks)} images -> {master_filename}")


def create_master_flats(raw_data_dir: str, processed_data_dir: str, data_method: str = "median", overwrite: bool = False):
    """Create master flats for all filters from the raw data dir and save it to the processed data dir"""
    # get a list of raw files
    raw_fits = ccdproc.ImageFileCollection(raw_data_dir)

    # get list of unique filters
    filter_list = set([CCDData.read(fits).meta.get("FILTER") for fits in list(raw_fits.files_filtered(IMTYPE="Sky", include_path=True))])
    log.debug(f"Filter List: {filter_list}")

    for flat_filter in filter_list:
        # find all unique exposure times in the flat set
        flat_times = set([CCDData.read(fits).meta.get("EXPTIME") for fits in list(raw_fits.files_filtered(IMTYPE="Sky", FILTER=flat_filter, include_path=True))])
        log.debug(f"flat_filter={flat_filter}\tflat_times={flat_times}")

        for exp_time in flat_times:
            # get all flats of the filter for the given exposure time
            flat_list = raw_fits.files_filtered(IMTYPE="Sky", FILTER=flat_filter, EXPTIME=exp_time, include_path=True)

            # define the master filename
            filter_str = str(flat_filter).strip().replace(" ", "_")
            master_filename = os.path.join(processed_data_dir, f"MasterFlat_{filter_str}.fits")

            # reduce each flat seperately in case there are multiple exposure times
            partial_master_list = []
            log.debug(f"Found flats {flat_list}")
            for fits in flat_list:
                partial_master = ccdproc.subtract_bias(CCDData.read(fits), get_master_calibration_file(processed_data_dir, IMTYPE="Bias"))
                partial_master = ccdproc.subtract_dark(partial_master, get_master_calibration_file(processed_data_dir, IMTYPE="Dark", EXPTIME=exp_time), dark_exposure=exp_time * units.second, data_exposure=exp_time * units.second)
                partial_master_list.append(partial_master)

            # TODO: Add flat scaling backup
            master_flat = ccdproc.combine(partial_master_list, method=data_method)
            if safe_write_ccddata(master_flat, master_filename, overwrite):
                log.info(f"Created Master Flat from {len(flat_list)} images -> {master_filename}")


def reduce_lights(raw_data_dir: str, processed_data_dir: str, overwrite: bool = False, rename_files: bool = False):
    """Reduce all lights in the raw data dir"""
    # get raw files
    raw_fits = ccdproc.ImageFileCollection(raw_data_dir).files_filtered(IMTYPE="Light")

    # reduce each fits file
    for fits in raw_fits:
        # get metadata from fits
        fits_ccd = CCDData.read(os.path.join(raw_data_dir, fits))
        target = fits_ccd.meta.get("OBJECT")
        exp_time = fits_ccd.meta.get("EXPTIME")
        filter = fits_ccd.meta.get("FILTER")
        file_number = str(fits).split("_")[-1].replace(".fits", "")

        # define master filename, option to rename the file based on header data (could be very destructive)
        if rename_files:
            master_filename = os.path.join(processed_data_dir, target, f"Reduced_{str(target).strip().replace(' ', '_')}_{exposure_to_str(exp_time)}_{str(filter).strip().replace(' ', '_')}_{file_number}.fits")
        else:
            master_filename = os.path.join(processed_data_dir, target, f"Reduced_{fits}.fits")

        # reduce and save fits
        log.debug(f"Attempting reduction of {fits}")
        reduced_fits = ccdproc.subtract_bias(fits_ccd, get_master_calibration_file(processed_data_dir, IMTYPE="Bias"))
        reduced_fits = ccdproc.subtract_dark(reduced_fits, get_master_calibration_file(processed_data_dir, IMTYPE="Dark", EXPTIME=exp_time), dark_exposure=exp_time * units.second, data_exposure=exp_time * units.second)
        reduced_fits = ccdproc.flat_correct(reduced_fits, get_master_calibration_file(processed_data_dir, IMTYPE="Sky", FILTER=filter))
        if safe_write_ccddata(reduced_fits, master_filename, overwrite):
            log.info(f"Reduced {fits} Target: {target} Exposure: {exp_time} Filter: {filter} -> {master_filename}")


if __name__ == "__main__":
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("RawDataDir", type=str,
                        help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("ProcessedDataDir", type=str,
                        help="Path to the folder you want saved data to go. (WARNING: May overwrite existing data)")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="True if you want to overwrite existing files with the same name. (default: false)")
    parser.add_argument("-n", "--name", action="store_true",
                        help="Renames files based on fits header data when doing light reduction.")
    args = parser.parse_args().__dict__

    # ensure theres is raw data to sort
    if not os.path.isdir(args["RawDataDir"]):
        log.error(f"User specified path {args['RawDataDir']} does not exist.")
        exit()

    # warn about options
    if args["name"]:
        log.warning(f"File Renaming is unstable and not guaranteed to work, it should only be used when all files are known to end in _###.fits and the header OBJECT is correct.")

    if args["overwrite"]:
        log.warning(f"Files will be overwritten.")

    # small wait for the user to potentially terminate the code before any of the above options take into effect
    wait_time = 5
    log.warning(f"Data reduction will begin in {wait_time}s. To abort: CTRL+C")
    time.sleep(wait_time)

    # run calibrate and reduce code
    raw_dir = Path(args["RawDataDir"]).absolute()
    processed_dir = Path(args["ProcessedDataDir"]).absolute()

    create_master_bias(raw_dir, processed_dir, overwrite=args["overwrite"])
    create_master_darks(raw_dir, processed_dir, overwrite=args["overwrite"])
    create_master_flats(raw_dir, processed_dir, overwrite=args["overwrite"])
    reduce_lights(raw_dir, processed_dir, overwrite=args["overwrite"], rename_files=args["name"])
