import argparse
import logging
import os
import shutil
import warnings
from pathlib import Path

import ccdproc

from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyWarning
from rich.logging import RichHandler

# suppresses the fits fixed warning (annoying)
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# configure loggers one for terminal one for file errors
log = logging.Logger(name="DataFilterLog")
formatter = logging.Formatter("%(name)s|%(asctime)s|[%(levelname)s]|:%(message)s")
log.setLevel(logging.INFO)

stream_handler = RichHandler()
stream_handler.setLevel(logging.DEBUG) # change this to change terminal readout
file_handler = logging.FileHandler(filename="filtersort.log", delay=True)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR) # change this to change what is logged to file

log.addHandler(stream_handler)
log.addHandler(file_handler)


def sort_fits(dir: str, keyword: str, delete_original: bool = True):
    for file in ccdproc.ImageFileCollection(dir).files_filtered():
        # get fits header and keyword entry
        fits = CCDData.read(os.path.join(dir, file)).copy()
        keyword_data = fits.meta.get(keyword)

        if keyword_data:
            keyword_dirname = os.path.join(dir, str(keyword_data).strip().replace(" ", "_").replace(".", "-").replace("/", "-").replace("\\", "-"))
            log.info(f"Found {keyword} -> Generated Save Path {keyword_dirname}")

            # create the save dir
            os.makedirs(keyword_dirname, exist_ok=True)

            try:
                # move file
                if delete_original:
                    log.info(f"Moved {file}")
                    shutil.move(os.path.join(dir, file), os.path.join(keyword_dirname, file))
                else:
                    log.info(f"Copied {file}")
                    shutil.copyfile(os.path.join(dir, file), os.path.join(keyword_dirname, file))
            except shutil.SameFileError:
                log.warning(f"Attempted to copy same file. Skipping. {file}")
        else:
            log.warning(f"Keyword {keyword} not found in {file}. Skipping.")
    log.info(f"All files moved from {dir}.")


if __name__ == "__main__":
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("DataDir", type=str,
                        help="Path to the folder containing your raw fits files. (Can be relative)")
    parser.add_argument("SortKeyword", type=str,
                        help="Keyword in FITS header to sort by.")
    parser.add_argument("-s", "--save-original", action="store_true",
                        help="Set to True to not delete original files.")
    args = parser.parse_args().__dict__

    # ensure theres is raw data to sort
    if not os.path.isdir(args["DataDir"]):
        log.error(f"User specified path {args['DataDir']} does not exist.")
        exit()

    # run calibrate and reduce code
    data_dir = Path(args["DataDir"]).absolute()
    keyword = args["SortKeyword"].strip().upper()
    sort_fits(data_dir, keyword, args["save_original"])
