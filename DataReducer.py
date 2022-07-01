import argparse
import logging
import os
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
log.setLevel(logging.DEBUG)

stream_handler = RichHandler()
stream_handler.setLevel(logging.DEBUG) # change this to change terminal readout
file_handler = logging.FileHandler(filename="debug.log", delay=True)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR) # change this to change what is logged to file

log.addHandler(stream_handler)
log.addHandler(file_handler)


class ImageReducer():
    def __init__(self, raw_data_dir: str, processed_data_dir: str) -> None:
        # define paths
        self.raw_data_path = Path(raw_data_dir).absolute()
        self.processed_data_path = Path(processed_data_dir).absolute()
        log.info(f"RawData: {self.raw_data_path}\tProcessedData: {self.processed_data_path}")

        # load all raw images found in target dir
        self.raw_images = ccdproc.ImageFileCollection(self.raw_data_path)
        log.info(f"Loaded {len(self.raw_images.files_filtered())} raw images")

        # place holders for master calibration files
        self.master_bias = None
        self.master_darks = None
        self.master_flats = None


    def create_master_bias(self, overwrite: bool = False) -> CCDData:
        """Create a master bias from all biases found in RawDataDir.

        Args:
            overwrite (bool, optional): Set to True to overwrite files in ProcessedDataDir. Defaults to False.

        Returns:
            CCDData: Returns the master bias.
        """

        # sort out biases from other images by IMTYPE
        raw_biases = self.raw_images.files_filtered(IMTYPE="Bias", include_path=True)

        # define the master file save path
        master_filename = os.path.join(self.processed_data_path, "MasterBias.fits")

        # combine and save biases using a median combine TODO: Make configurable
        self.master_bias = ccdproc.combine(raw_biases, method="median")

        # catch the file already exists error
        self._safe_write_ccddata(self.master_bias, master_filename, overwrite)

        log.info(f"Created Master Bias from {len(raw_biases)} images -> {master_filename}")
        return self.master_bias


    def create_master_darks(self, overwrite: bool = False) -> dict:
        """Create darks for each set of exposure times found in RawDataDir.

        Args:
            overwrite (bool, optional): Set to True to overwrite files in ProcessedDataDir. Defaults to False.

        Returns:
            dict: Returns a dict of exposure time to master dark for flat division and raw reduction.
        """

        # ensure master biases have been made
        if not self.master_bias:
            log.warning("Create master darks run before create master biases.")
            self.create_master_bias()

        # find all unique exposure times
        dark_times = set([CCDData.read(a).meta.get("EXPTIME") for a in list(self.raw_images.files_filtered(IMTYPE="Dark", include_path=True))])

        # prepare output dict
        self.master_darks = {}

        # each unique time gets a master dark which is saved to the dictionary
        for time in dark_times:
            # sort out darks of a specific exposure from other images by IMTYPE and EXPTIME
            selected_darks = self.raw_images.files_filtered(EXPTIME=time, IMTYPE="Dark", include_path=True)

            # define the master file save path
            save_time = self._exp_time_to_str(time)
            master_filename = os.path.join(self.processed_data_path, f"MasterDark{save_time}.fits")

            # combine and save darks using a median combine
            master_dark = ccdproc.combine(selected_darks, method="median")

            # catch the file exists error when writing and log -> exit
            self._safe_write_ccddata(master_dark, master_filename, overwrite)

            # save the master dark to the dark dictionary
            self.master_darks[save_time] = CCDData.read(master_filename)

            log.info(f"Created {save_time} Master Dark from {len(selected_darks)} images -> {master_filename}")
        return self.master_darks


    def create_master_flats(self, overwrite: bool = False) -> dict:
        """Create master flats for each filter if there exists a dark with the same exposure time. Only accepts specific filters.

        Args:
            overwrite (bool, optional): Set to True to overwrite files in ProcessedDataDir. Defaults to False.

        Returns:
            dict: Returns a dict of filter to master_flat for raw reduction.
        """

        # ensure master darks (and therefore master biases) have been created
        if not self.master_darks:
            log.warning("Create master flats run before create master darks.")
            self.create_master_darks()

        # get all filters used
        filters = self._get_unique_set("Sky", "FILTER")
        log.info(f"Found flats in the following filters: {filters}")

        # create master flats
        self.master_flats = {}

        for flat_filter in filters:
            # list all flats of the filter
            flat_list = self.raw_images.files_filtered(IMTYPE="Sky", FILTER=flat_filter, include_path=True)

            # find all unique exposure times in the flat files
            flat_times = self._get_unique_set("SKY", "EXPTIME")

            # checks for uniformity in flat times and skips non-uniform flat collections
            if len(flat_times) != 1:
                log.error(f"Flats of the same filter ({flat_filter}) taken with multiple exposures, found {flat_times} exposures. Skipping.")
                continue

            # since flat times are uniform after the above step flat_time and flat_time[0] are the same (len 1 array)
            flat_time = self._exp_time_to_str(flat_times[0])
            # define the master save location
            master_filename = os.path.join(self.processed_data_path, f"MasterFlat{flat_filter}.fits")

            # TODO: Add scaling backup
            # create the master flat with a median combine, bias subtraction, and dark subtraction from master files
            master_flat = ccdproc.combine(flat_list, method="median")
            master_flat = ccdproc.subtract_bias(master_flat, self.master_bias)
            master_flat = ccdproc.subtract_dark(master_flat, self.master_darks[flat_time],
                                                dark_exposure=flat_time * units.second, data_exposure=flat_time * units.second)
            self._safe_write_ccddata(master_flat, master_filename, overwrite)

            # save the master flat to the flat dictionary
            self.master_flats[flat_filter] = CCDData.read(master_filename)

            log.info(f"Created {flat_filter} {flat_time} Master Flat from {len(flat_list)} images -> {master_filename}")
        return self.master_flats


    def create_calibration_images(self, overwrite: bool = False) -> None:
        """Creates master biases, darks, and flats.

        Args:
            overwrite (bool, optional): Set to True to overwrite files in ProcesedDataDir. Defaults to False.
        """

        self.create_master_bias(overwrite=overwrite)
        self.create_master_darks(overwrite=overwrite)
        self.create_master_flats(overwrite=overwrite)


    def reduce_raw_lights(self, overwrite: bool = False) -> None:
        """Reduce files of IMTYPE=Light found in the RawDataDir

        Args:
            overwrite (bool, optional): Set to True if you want to overwrite files in ProcessedDataDir. Defaults to False.
        """

        # get all unique objects
        objects_list = self._get_unique_set("Light", "Object", True)

        for obj_observed in objects_list:
            # get the observed object for creating the save dir and sorting images
            obj_name = obj_observed.strip().replace(" ", "") # TODO: filter *-/\.
            obj_save_dir = os.path.join(self.processed_data_path, obj_name)

            for image in self.raw_images.files_filtered(IMTYPE="Light", OBJECT=obj_observed):
                # read the light as ccddata
                ccd_image = CCDData.read(os.path.join(self.raw_data_path, image))

                # define the save path
                reduced_filename = os.path.join(obj_save_dir, "Reduced_" + image)

                # get exposure and filter from ccd data to access the dark/filters dicts
                exposure_time = ccd_image.meta.get("EXPTIME")
                exposure_string = self._exp_time_to_str(exposure_time)
                image_filter = ccd_image.meta.get("FILTER")

                # ensure the exposure time has been processed
                if exposure_string not in self.master_darks: # TODO: Add dark scaling as backup
                    log.error(f"No dark for the exposure taken ({exposure_string}). Light: {image}")
                    continue

                # ensure the filter has been processed
                if image_filter not in self.master_flats:
                    log.error(f"No flat for the filter taken ({image_filter}). Light: {image}")
                    continue

                # create a reduced light by subtracting bias, darks, and doing a flat division
                reduced = ccdproc.subtract_bias(ccd_image, self.master_bias)
                reduced = ccdproc.subtract_dark(reduced, self.master_darks[exposure_string],
                                                dark_exposure=exposure_time * units.second, data_exposure=exposure_time * units.second)
                reduced = ccdproc.flat_correct(reduced, self.master_flats[image_filter])
                reduced.write(reduced_filename, overwrite=overwrite)

                log.info(f"Reduced {image} using {image_filter} Filter {exposure_string} Exposure {obj_name} -> {reduced_filename}")


    def _safe_write_ccddata(self, fits: CCDData, filename: str, overwrite: bool = False) -> CCDData:
        """Atttempt to write a .fits file and handle errors.

        Args:
            overwrite (bool, optional): Set to True to overwrite a .fits that exists. Defaults to False.

        Returns:
            CCDData: Return the .fits file.
        """

        try:
            os.makedirs(filename, exist_ok=True)
            fits.write(filename, overwrite=overwrite)
        except OSError as e:
            log.error(f"File {filename} already exists and overwrite={overwrite}.")
            exit()
        return fits

    def _exp_time_to_str(self, exposure_time) -> str:
        """Convert an exposure time to string, used to normalize between formats"""
        if int(exposure_time) == exposure_time:
            # checks for an integer exposure time greater than 1
            return str(int(exposure_time)) + "s"
        elif int(exposure_time*1000) == exposure_time*1000:
            # checks for an in integer exposure time in milliseconds
            return str(int(exposure_time*1000)) + "ms"
        return str(exposure_time)

    def _get_unique_set(self, IMTYPE: str, fits_tag: str, include_path: bool = False) -> list:
        return list(set([CCDData.read(f).meta.get(fits_tag) for f in self.raw_images.files_filtered(IMTYPE=IMTYPE, include_path=include_path)]))


if __name__ == "__main__":
    """
    __FILE NAME FORMAT__:
    {ImageType}_{Filter}_{ExposureTime}_{FileNumber}.fits
    __FILTER NAME FORMAT__:
    - Red = R
    - Blue = B
    - Green = G
    - Bessel R = BR
    - Bessel B = BB
    - Bessel G = BG
    - None = CLEAR
    - OIII = O3
    - SII = S2
    - Infrared = I
    - Bessel Infrared = BI
    - UV = UV
    - Bessel UV = BUV
    - H Beta = HB
    - H Alpha = HA
    """

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("RawDataDir", type=str,
                        help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("ProcessedDataDir", type=str,
                        help="Path to the folder you want saved data to go. (WARNING: May overwrite existing data)")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="True if you want to overwrite existing files with the same name. (default: false)")
    args = parser.parse_args().__dict__

    # ensure theres is raw data to sort
    if not os.path.isdir(args["RawDataDir"]):
        log.error(f"User specified path {args['RawDataDir']} does not exist")
        exit()

    # run calibrate and reduce code
    reducer = ImageReducer(args["RawDataDir"], args["ProcessedDataDir"])
    reducer.create_calibration_images(overwrite=args["overwrite"])
    reducer.reduce_raw_lights()