# -*- coding: utf-8 -*-
"""Best-effort DICOM deidentification.
Resources:

Mandatory DICOM tags
https://www.pclviewer.com/help/required_dicom_tags.htm

Deidentification requirements
https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup55.pdf
(Table X.1-1)
"""
import os.path

from deid.config import DeidRecipe
from deid.dicom.parser import DicomParser
import pydicom
from pydicom.uid import generate_uid
from pydicom.errors import InvalidDicomError
from glob import glob
import logging
import multiprocessing as mp
from typing import List
import importlib.resources


class DicomDeidentifier:
    """
    This class allows to pseudonymize an instance of
    pydicom.Dataset with our custom recipe and functions.
    """

    def __init__(
        self,
        recipe: str | List[str] = None,
        verbose: bool = None,
        out_path: str = None,
        processes: int = 1,
    ) -> None:
        """
        :param recipe: path to our deid recipe.
        """

        recipe = [
            f"{importlib.resources.files('mede')}/dicom_deid/deid_options/{option}.dicom"
            for option in recipe
        ]
        self.recipe: DeidRecipe = DeidRecipe(recipe)
        self._processes: int = processes
        self._verbose: bool = verbose if verbose is not None else False
        self._out: str = out_path
        if self._verbose:
            logging.info(f"Using recipe(s) {recipe} for DICOM deidentification")

    def __call__(self, dataset: str, out: str = None) -> None:
        """Pseudonymize a single dicom dataset

        :param dataset: dataset that will be pseudonymized
        :returns: pseudonymized dataset
        """
        if os.path.isfile(dataset):
            new_name = dataset.split(".")[0] + "_deidentified.dcm"
            try:
                dcm = pydicom.dcmread(dataset)  # Consider stop_before_pixels
            except InvalidDicomError:
                raise InvalidDicomError(
                    f"File is not valid dicom, specify --force-dicom to force reading"
                )
            parser = DicomParser(dcm, self.recipe)
            # register functions that are specified in the recipe
            parser.define("new_UID", self.new_UID)
            # parse the dataset and apply the deidentification
            parser.parse()
            parser.dicom.save_as(new_name)
            if self._verbose:
                logging.info(f"Saved\t{new_name}")

        elif os.path.isdir(dataset):
            if self._out is None:
                self._out = dataset
            os.makedirs(self._out, exist_ok=True)
            found_files = glob(os.path.join(dataset, "**/*"), recursive=True)
            with mp.Pool(processes=self._processes) as process_pool:
                process_pool.starmap(
                    self._deidentify,
                    [
                        (file, self.recipe, self._out, self.new_UID)
                        for file in found_files
                    ],
                )
        else:
            raise TypeError(f"{dataset} is neither a directory nor a file!")

    @staticmethod
    def _deidentify(dcm_file: str, recipe, out, func):
        if os.path.isdir(dcm_file):
            return
        filename = dcm_file.split(os.path.sep)[-1]
        try:
            dcm = pydicom.dcmread(dcm_file)
        except InvalidDicomError:
            raise InvalidDicomError(
                f"File is not valid dicom, specify --force-dicom to force reading"
            )
        parser = DicomParser(dcm, recipe)
        parser.define("new_UID", func)
        parser.parse()
        if filename.lower().endswith('.dcm'):
            base = '.'.join(filename.split('.')[:-1])
            new_fn = generate_uid(entropy_srcs=[base]) + '.dcm'
            new_fn = os.path.join(out, new_fn)
        elif filename.lower() == 'dicomdir':
            return
        else:
            new_fn = os.path.join(out, filename + "_deidentified")

        parser.dicom.save_as(new_fn)

    # All registered functions that are used in the recipe must
    # receive the arguments: `item`, `value`, `field`, `dicom`

    @staticmethod
    def new_UID(item, value, field, dicom) -> str:
        """Generate new UID based on the old one it stays consistent"""
        return generate_uid(entropy_srcs=[field.element.value])