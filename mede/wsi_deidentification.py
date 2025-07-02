import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import logging
import os
import numpy as np
from glob import glob
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)


class WSIDeidentifier:
    """
    Class for deidentifying the image information of whole slide images (WSI).

    Args:
        verbose (bool, optional): If True, enables verbose logging. Defaults to False.
        out_path (str, optional): The output path for the deidentified images. Defaults to None.

    Methods:
        __call__(self, dataset: str) -> None:
            Deidentifies the specified dataset by calling the _deidentify method.

        _deidentify(dcm_file: str, out: str, verbose: bool = False) -> None:
            Deidentifies a DICOM file by anonymizing the scan label and overview (if present).
            Saves the deidentified file to the specified output path.
    """

    def __init__(self, verbose: bool = False, out_path: str = None) -> None:
        self._verbose = verbose if verbose is not None else False
        self._out = out_path

    def __call__(self, dataset: str) -> None:
        """
        Deidentifies the specified dataset by calling the _deidentify method.

        Args:
            dataset (str): The path to the dataset to be deidentified.

        Returns:
            None
        """
        if self._verbose:
            logging.info(f"Deidentifying {dataset}")
        if os.path.isdir(dataset):
            for dcm in glob(f"{dataset}/*.dcm"):
                self._deidentify(dcm, self._out, self._verbose)
        elif os.path.isfile(dataset):
            self._deidentify(dataset, self._out, self._verbose)

    @staticmethod
    def _deidentify(dcm_file: str, out: str, verbose: bool = False) -> None:
        """
        Deidentifies a DICOM file by anonymizing the scan label and overview (if present).
        Saves the deidentified file to the specified output path.

        Args:
            dcm_file (str): The path to the DICOM file to be deidentified.
            out (str): The output path for the deidentified file.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.

        Returns:
            None
        """
        dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)

        # Anonymize scan label
        if "LABEL" in dcm[0x0008, 0x0008].value:
            dcm = pydicom.dcmread(dcm_file)
            try:
                dcm.PixelData = np.zeros_like(dcm.pixel_array).tobytes()
                dcm.save_as(os.path.join(out, Path(dcm_file).name))
                if verbose:
                    logging.info(f"Label file anonymized!")
            except:
                logging.info("Label file does not contain pixel array, remove file instead of overwriting ...")
                if os.path.exists(os.path.join(out, Path(dcm_file).name)):
                   os.remove(os.path.join(out, Path(dcm_file).name))

        # Anonymize overview
        elif "OVERVIEW" in dcm[0x0008, 0x0008].value:
            dcm = pydicom.dcmread(dcm_file)
            dcm.pixel_array[:, :250, :] = 0
            dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dcm.PixelData = dcm.pixel_array.tobytes()
            dcm.save_as(os.path.join(out, Path(dcm_file).name))
            if verbose:
                logging.info(f"Overview file anonymized!")
        else:
            # Copy the file if it doesn't exist in the output folder
            if not os.path.exists(os.path.join(out, Path(dcm_file).name)):
                shutil.copy2(dcm_file, os.path.join(out, Path(dcm_file).name))
