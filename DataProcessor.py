import argparse
from doctest import master
from functools import reduce
from xml.etree.ElementInclude import include
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
    try:
        os.makedirs(filename.parent, exist_ok=True)
        fits.write(filename, overwrite=overwrite)
    except OSError as e: pass


def exp_to_str(exposure_time: float) -> str:
    """ Convert an exposure time to a string with units """
    if int(exposure_time) == exposure_time:
        return str(int(exposure_time)) + "s"
    elif int(exposure_time*1000) == exposure_time*1000:
        return str(int(exposure_time*1000)) + "ms"
    return str(exposure_time)


# calibration images class
class CalibrationImages:
    def __init__(self, raw_dir: Path, master_dir: Path, overwrite: bool = False, program_flag: str = "DATAPROCESSOR"):
        self.raw_dir = raw_dir
        self.master_dir = master_dir
        self.overwrite = overwrite
        self.program_flag = program_flag

        # place holders for masters
        self.master_bias = None
        self.master_dark_dict = None
        self.master_flat_dict = None

        # raw fits files
        self.raw_fits = ccdproc.ImageFileCollection(raw_dir)


    def create_master_bias(self, method: str = "median", filename: str = "MasterBias.fits") -> Path:
        """ Create a master bias from raw biases and saved to master dir, return a path to the file"""
        bias_fits = self.raw_fits.files_filtered(IMTYPE="Bias", include_path=True)

        # combine bias fits
        master_bias = ccdproc.combine(bias_fits, method=method)

        # add program flag
        master_bias.meta[self.program_flag] = True

        # save and return
        master_bias_path = Path(self.master_dir, filename)
        safe_write_ccddata(master_bias_path, master_bias, self.overwrite)
        self.master_bias = master_bias_path

        log.info(f"Created master bias from {len(bias_fits)} files > {master_bias_path}")
        return master_bias_path


    def create_master_darks(self, method: str = "median", filename: str = "MasterDark") -> dict:
        """ Create master darks for each dark time found in raw dir and save the master dir, return a dict of paths"""
        dark_fits = self.raw_fits.files_filtered(IMTYPE="Dark", include_path=True)

        # get all unique dark times
        dark_times = set([CCDData.read(fits).meta.get("EXPTIME") for fits in dark_fits])

        # init the dark dict
        dark_dict = {}

        # loop over unique dark times to create a master dark for each
        for exp_time in tqdm(dark_times, desc="Creating Master Darks"):
            selected_darks = self.raw_fits.files_filtered(IMTYPE="Dark", EXPTIME=exp_time, include_path=True)

            # combine selected darks
            master_dark = ccdproc.combine(selected_darks, method=method)

            # add program flag
            master_dark.meta[self.program_flag] = True

            # save
            master_dark_path = Path(self.master_dir, f"{filename}_{exp_to_str(exp_time)}.fits")
            safe_write_ccddata(master_dark_path, master_dark, self.overwrite)
            dark_dict[exp_to_str(exp_time)] = master_dark_path

        log.info(f"Created master darks for times {dark_times} > {self.master_dir}")
        self.master_dark_dict = dark_dict
        return dark_dict


    def create_master_flat(self, method: str = "median", filename: str = "MasterFlat") -> dict:
        """ Create master flats for each filter in raw dir and save to master dir, return a dict of paths"""
        flat_fits = self.raw_fits.files_filtered(IMTYPE="Sky", include_path=True)

        # get unique flat filters
        flat_filters = set([CCDData.read(fits).meta.get("FILTER") for fits in flat_fits])

        # init flat dict and masters and loop over each filter
        flat_dict = {}
        master_bias = CCDData.read(self.master_bias)
        master_darks = {k:CCDData.read(v) for k, v in tqdm(self.master_dark_dict.items(), desc="Loading Darks")}
        for flat_filter in tqdm(flat_filters, desc="Creating Flats"):
            selected_flats = self.raw_fits.files_filtered(IMTYPE="Sky", FILTER=flat_filter, include_path=True)

            # read and dark subtract each flat fit (could have a unique time)
            # TODO: Add dark scaling
            fit_list = []
            for fit in selected_flats:
                fit = CCDData.read(fit)
                fit_exposure = fit.meta.get("EXPTIME")

                # bias and dark subtract
                fit = ccdproc.subtract_bias(fit, master_bias)
                fit = ccdproc.subtract_dark(fit, master_darks[exp_to_str(fit_exposure)],
                                            dark_exposure=fit_exposure*units.s, data_exposure=fit_exposure*units.s)

                # add fit to fit list for combining
                fit_list.append(fit)

            # combine fits and save
            master_flat = ccdproc.combine(fit_list, method=method)
            master_flat.meta[self.program_flag] = True

            master_flat_filename = Path(self.master_dir, f"{filename}_{flat_filter}.fits")
            safe_write_ccddata(master_flat_filename, master_flat, self.overwrite)
            flat_dict[flat_filter] = master_flat_filename

        log.info(f"Created master flats for filters {flat_filters} > {self.master_dir}")
        self.master_flat_dict = flat_dict
        return flat_dict


    def find_calibration_images(self, only_with_flag: bool = True) -> tuple[Path, dict, dict]:
        """ Find calibration images in the master dir and return a tuple of bias, darks, flats"""
        master_fits = ccdproc.ImageFileCollection(self.master_dir)

        # get master bias
        if only_with_flag: master_bias_fits = master_fits.files_filtered(IMTYPE="Bias", include_path=True, **{self.program_flag:True})
        else: master_bias_fits = master_fits.files_filtered(IMTYPE="Bias", include_path=True)

        if len(master_bias_fits) == 1: self.master_bias = master_bias_fits[0]
        # for now just always use the first
        # TODO: add some selction criteria
        elif len(master_bias_fits) > 1: self.master_bias = master_bias_fits[0]

        # get master darks
        if only_with_flag: master_dark_fits = master_fits.files_filtered(IMTYPE="Dark", include_path=True, **{self.program_flag:True})
        else: master_dark_fits = master_fits.files_filtered(IMTYPE="Dark", include_path=True)

        self.master_dark_dict = {}
        for m_dark in master_dark_fits:
            m_dark_fit = CCDData.read(m_dark)
            self.master_dark_dict[exp_to_str(m_dark_fit.meta.get("EXPTIME"))] = m_dark

        # get master flats
        if only_with_flag: master_flat_fits = master_fits.files_filtered(IMTYPE="Sky", include_path=True, **{self.program_flag:True})
        else: master_flat_fits = master_fits.files_filtered(IMTYPE="Sky", include_path=True, **{self.program_flag:True})

        self.master_flat_dict = {}
        for m_flat in master_flat_fits:
            m_flat_fit = CCDData.read(m_dark)
            self.master_flat_dict[m_flat_fit.meta.get("FILTER")] = m_flat

        return (self.master_bias, self.master_dark_dict, self.master_flat_dict)


    def get_calibration_images(self) -> tuple[Path, dict, dict]:
        """ Return the master bias, master dark dict, and master flat dict """
        return (self.master_bias, self.master_dark_dict, self.master_flat_dict)


    def create_missing_images(self):
        """ determine which master calibration images are missing and create them """
        images = self.find_calibration_images
        if not images[0]: self.create_master_bias
        if not images[1]: self.create_master_darks
        if not images[2]: self.create_master_flat


    def reduce_lights(self, reduced_data_dir: Path, object_sort: bool = False, filter_sort: bool = False):
        """ Reduce all lights found in the data dir and move the reduced files to the reduced data dir, optional sorting methods"""
        light_fits = self.raw_fits.files_filtered(IMTYPE="Light", include_path=True)

        # get masters
        master_bias = CCDData.read(self.master_bias)
        master_darks = {k:CCDData.read(v) for k, v in tqdm(self.master_dark_dict.items(), desc="Loading Darks")}
        master_flats = {k:CCDData.read(v) for k, v in tqdm(self.master_flat_dict.items(), desc="Loading Flats")}

        # iterate over every fit and reduce and resave
        for fit in tqdm(light_fits, desc="Reducing lights"):
            fit_ccd = CCDData.read(fit)
            exp_time = fit_ccd.meta.get("EXPTIME")
            exp_str = exp_to_str(exp_time)
            fit_filter = fit_ccd.meta.get("FILTER")

            # reduce fit
            reduced_fit = ccdproc.subtract_bias(fit_ccd, master_bias)
            reduced_fit = ccdproc.subtract_dark(reduced_fit, master_darks[exp_str],
                                                dark_exposure=exp_time*units.s, data_exposure=exp_time*units.s)
            reduced_fit = ccdproc.flat_correct(reduced_fit, master_flats[fit_filter])

            # save
            fit_path_list = [reduced_data_dir]
            if object_sort: fit_path_list.append(fit_ccd.meta.get("OBJECT"))
            if filter_sort: fit_path_list.append(fit_ccd.meta.get("FILTER"))
            fit_path_list.append(os.path.basename(fit))

            fit_path = Path(*fit_path_list)
            safe_write_ccddata(fit_path, reduced_fit, self.overwrite)






# TODO: Add logging
if __name__ == "__main__":
    """ Run data reducer to.. reduce the data """
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=str, help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("reduced_dir", type=str, help="Path to the folder containing reduced data.")
    parser.add_argument("-m", "--master-dir", type=str, default=None, help="Specify a seperate path to save master calibration images")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing files when creating calibration images and saving.")
    parser.add_argument("-t", "--target-sort", action="store_true", help="Sort reduced lights by object")
    parser.add_argument("-f", "--filter-sort", action="store_true", help="Sort reduced lights by filter (object > filter) if both")
    parser.add_argument("-c", "--create-calibration", action="store_true", help="Create calibration images")
    parser.add_argument("-l", "--lights", action="store_true", help="Reduce lights")

    args = parser.parse_args().__dict__

    # arg processing
    if not args["master_dir"]: args["master_dir"] = args["reduced_dir"]

    # create a reducer object
    reducer = CalibrationImages(args["raw_dir"], args["master_dir"], args["overwrite"])

    # get or create calibration images
    if args["create_calibration"]:
        log.info("Creating calibration images")
        reducer.create_master_bias()
        reducer.create_master_darks()
        reducer.create_master_flat()
    else:
        log.info("Loading calibration images")
        bias, dark, flat = reducer.find_calibration_images()
        log.info(f"{bias}\n{dark}\n{flat}")

    # reduce lights
    if args["lights"]:
        log.info("Reducing lights")
        reducer.reduce_lights(args["reduced_dir"], args["target_sort"], args["filter_sort"])