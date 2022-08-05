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
from multiprocessing import cpu_count, Pool
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
def safe_write_ccddata(fits: CCDData, filename: str, overwrite: bool = False):
    """Writes .fits files and handles file exists errors"""
    # attempt to write and catch file exist error
    try:
        # find dirname and attempt creation
        dir_name = Path(filename).parent.absolute()
        log.debug(f"Checking {fits} parent dir {dir_name}. overwrite={overwrite}")
        os.makedirs(dir_name, exist_ok=True)

        # data tag to alert program of step
        fits.meta["REDUCER_PROCESSED"] = True
        fits.write(filename, overwrite=overwrite)
    except OSError as e:
        log.warning(f"File {filename} already exists. overwrite={overwrite}")
        log.warning(e)


def exposure_to_str(exposure_time) -> str:
    """Converts an exposure double to readable string"""
    if int(exposure_time) == exposure_time:
        # checks for an integer exposure time greater than 1
        return str(int(exposure_time)) + "s"
    elif int(exposure_time*1000) == exposure_time*1000:
        # checks for an in integer exposure time in milliseconds
        return str(int(exposure_time*1000)) + "ms"
    return str(exposure_time)


def get_master_bias(master_dir: Path) -> Path:
    """ get a master bias file from the master dir """
    master_fits = ccdproc.ImageFileCollection(master_dir)
    return master_fits.files_filtered(IMTYPE="Bias", include_path=True)[0]


def get_master_darks(master_dir: Path) -> dict:
    """ return the master dark dict """
    master_fits = ccdproc.ImageFileCollection(master_dir)
    master_darks = master_fits.files_filtered(IMTYPE="Dark", include_path=True)
    dark_dict = {}
    for dark in master_darks:
        dark_fit = CCDData.read(dark)
        dark_dict[exposure_to_str(dark_fit.meta.get("EXPTIME"))] = dark
    return dark_dict


def get_master_flats(master_dir: Path) -> dict:
    """ Return a master flat dict """
    master_fits = ccdproc.ImageFileCollection(master_dir)
    master_flats = master_fits.files_filtered(IMTYPE="Sky", include_path=True)
    flat_dict = {}
    for flat in master_flats:
        flat_fit = CCDData.read(flat)
        flat_dict[flat.meta.get("FILTER")] = flat
    return flat_dict


def create_master_calibrations(raw_dir: Path, master_dir: Path, method: str = "median",
                               overwrite: bool = False, process_count: int = 1,
                               do_bias: bool = True, do_darks: bool = True, do_flats: bool = True):
    """ Create the master calibration images, can be configured for creating only some master images """
    def _combine(filename: str, fits_list: list) -> Path:
        """ Wrapper for combine because I cant pass named args """
        master = ccdproc.combine(fits_list, method=method)
        safe_write_ccddata(master, filename, overwrite)
        return filename


    def _create_flat(flat_list: list, master_darks: dict, master_bias: Path):
        """ Wrapper for creating a flat to multiprocess """
        subtracted_flat_list = []

        # subtract bias and dark from each flat in list
        for flat in flat_list:
            flat_fit = CCDData.read(flat)
            flat_fit = ccdproc.subtract_bias(flat_fit, CCDData.read(master_bias))

            # get exptime and check if its in the dark dict
            exp_time = flat.meta.get("EXPTIME")
            if exp_time in master_darks:
                # exptime exists so subtract bias and add the result to the subtracted flat list
                master_dark = CCDData.read(master_darks[exposure_to_str(exp_time)])
                # TODO: Add scaling
                flat_fit = ccdproc.subtract_dark(flat_fit, master_dark, dark_exposure=exp_time * units.second, data_exposure=exp_time * units.second)
                subtracted_flat_list.append(flat_fit)
            # exptime not found, skip just that file but attempt master flat creation with the rest
            else: log.warning(f"Exposure time {exp_time} not found in master darks list. Skipping flat {flat}")

        # check if enough flats were created to make a master flat (min 3)
        if len(subtracted_flat_list) > 3:
            # combine, write, and return the master flat
            master_flat = ccdproc.combine(subtracted_flat_list, method=method)
            safe_write_ccddata(master_flat)
            return master_flat
        # not enough flats, skip processing
        else:
            # take a guess at filter
            filter = CCDData.read(flat_list[0].meta.get("FILTER"))
            log.error(f"Not enough flats were bias/dark subtracted to create a master (min 3). Skipping filter {filter}.")


    # get a list of all raw files
    raw_fits = ccdproc.ImageFileCollection(raw_dir)
    master_fits = ccdproc.ImageFileCollection(master_dir)

    # get all unique exposure times for darks
    dark_fits = raw_fits.files_filtered(IMTYPE="Dark", include_path=True)
    dark_times = set([CCDData.read(fits).meta.get("EXPTIME") for fits in dark_fits])

    # get all unique filters for flats
    flat_fits = raw_fits.files_filtered(IMTYPE="Sky", include_path=True)
    flat_filters = set([CCDData.read(fits).meta.get("FILTER") for fits in flat_fits])

    # create process pool
    with Pool(processes=process_count) as pool:
        # check for bias creation
        if do_bias:
            bias_name = Path(master_dir, "MasterBias.fits")
            bias_fits = raw_fits.files_filtered(IMTYPE="Bias", include_path=True)
            bias_process = pool.apply_async(_combine, (bias_name, bias_fits))

        # check for dark creation
        if do_darks:
            dark_process_list = []
            for exp_time in tqdm(dark_times, desc="Creating Master Darks"):
                # get darks of specific exposure
                selected_darks = raw_fits.files_filtered(IMTYPE="Dark", EXPTIME=exp_time, include_path=True)

                # apply it to process
                filename = Path(master_dir, f"MasterDark_{exposure_to_str(exp_time)}.fits")
                dark_process_list.append(pool.apply_async(_combine, (filename, selected_darks)))

        # check for flat creation
        if do_flats:
            flat_process_list = []

            # get master bias
            master_bias = None
            # TODO: Fix pickling (I hate pickle) perhaps just use a while loop to wait until done?
            if do_bias: master_bias = bias_process.get()
            else: master_bias = get_master_bias(master_dir)

            # get master darks list
            if do_darks:
                # if they were just made
                for dark_process in dark_process_list: dark_process.get()
            master_darks = get_master_darks(master_dir)

            for flat_filter in tqdm(flat_filters, desc="Creating Master Flats"):
                # get flats of current filter
                selected_flats = raw_fits.files_filtered(IMTYPE="Sky", FILTER=flat_filter, include_path=True)

                # append filter to process
                flat_process_list.append(pool.apply_async(_create_flat, (selected_flats, master_darks, master_bias)))

            # collect flat processes
            for flat_process in flat_process_list: flat_process.get()
        else:
            # preform process collection if flats arent being done
            if do_bias: bias_process.get()

            if do_darks:
                for dark_process in dark_process_list: dark_process.get()


def reduce_lights(raw_dir: Path, master_dir: Path, reduced_dir: Path, overwrite: bool = False, process_count: int = 1):
    """ Apply data reduction to lights and sort into folders based on object """
    def _reduce_light(light: Path, bias: Path, dark: Path, flat: Path, filename: str):
        """ Reduce a given light """
        light_fit = CCDData.read(light)
        light_fit = ccdproc.subtract_bias(light_fit, CCDData.read(bias))

        # master dark subtract
        dark_fit = CCDData.read(dark)
        dark_time = dark_fit.meta.get("EXPTIME")
        # TODO: Add scaling
        light_fit = ccdproc.subtract_dark(light_fit, dark_fit, dark_exposure=dark_time * units.second, data_exposure=dark_time * units.second)

        # flat divide
        light_fit = ccdproc.flat_correct(light_fit, CCDData.read(flat))

        # save
        safe_write_ccddata(light_fit, filename, overwrite)
        return filename

    # get a list of all raw files
    raw_fits = ccdproc.ImageFileCollection(raw_dir)

    # get master lists
    master_bias = get_master_bias(master_dir)
    master_darks = get_master_darks(master_dir)
    master_flats = get_master_flats(master_dir)

    # iterate through lights and send each one to new process for reduction
    with Pool(processes=process_count) as pool:
        light_processes = []
        for light in raw_fits.files_filtered(IMTYPE="Light", include_path=True):
            # read fits
            light_fit = CCDData.read(light)

            # get properties
            light_filter = light_fit.meta.get("FILTER")
            light_time = light_fit.meta.get("EXPTIME")

            # get masters
            master_dark = master_darks[exposure_to_str(light_time)]
            master_flat = master_flats[light_filter]

            # create process
            filename = Path(reduced_dir, f"Reduced_{os.path.basename(light)}")
            light_processes.append(pool.apply_async(_reduce_light, (light, master_bias, master_dark, master_flat, filename)))

    # catch processes
    for process in light_processes: process.get()


# TODO Make a combine lights option to combine like images (challenge is choosing which metrics to sort to combine)
if __name__ == "__main__":
    """ Run data reducer to.. reduce the data """
    # TODO: add multiprocessing

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("RawDir", type=str,
                        help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("ProcessedDir", type=str,
                        help="Path to the folder you want saved data to go. (WARNING: May overwrite existing data)")
    parser.add_argument("-m", "--master-dir", type=str, default=None,
                        help="A seperate dir for master calibration file saving.")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="True if you want to overwrite existing files with the same name. (default: false)")
    args = parser.parse_args().__dict__

    # ensure theres is raw data to sort
    if not os.path.isdir(args["RawDir"]):
        log.error(f"User specified path {args['RawDir']} does not exist.")
        exit()

    # process args
    raw_dir = Path(args["RawDir"])
    processed_dir = Path(args["ProcessedDir"])
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # check for master dir arg
    if args["master_dir"]: master_dir = Path(args["MasterDir"])
    else: master_dir = processed_dir
    os.makedirs(master_dir, exist_ok=True)

    # run processing code
    create_master_calibrations(raw_dir, master_dir, args["overwrite"], cpu_count())
    reduce_lights(raw_dir, processed_dir, args["overwrite"], cpu_count())

