# -*- coding: utf-8 -*-
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from dataset import get_inference_loader, resample, dcm2nifti, nifti2dcm
import numpy as np
from torchvision.transforms import v2
import nibabel as nib
import time
import twixtools
from glob import glob

logging.basicConfig(level=logging.INFO)

class Inference:
    """
    Class for performing inference on input data and saving the output data.

    Args:
        output_path (str): The path to save the output data.
        gpu (int, optional): The GPU device index to use for inference. Defaults to None (CPU).
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        skullstrip (bool, optional): Whether to perform skull stripping. Defaults to False.
        deface (bool, optional): Whether to perform defacing. Defaults to False.
    """

    def __init__(self,
                 output_path: str,
                 gpu: int = None,
                 verbose: bool = False,
                 skullstrip: bool = False,
                 deface: bool = False,
                 ) -> None:
        self.input_path = None
        self.output_path = output_path
        self.model_path = "model/kStrip.pth"
        self.stem = "skullstrip"  # stem for output file
        self._verbose = verbose if verbose is not None else False
        if gpu is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
            )

    def __call__(
            self,
            input_path: str,
    ) -> None:
        """
        Perform inference on the input data.

        Args:
            input_path (str): The path to the input data.
        """
        self.input_path = input_path
        self.run()
        
    def load_twix(self, filename: str) -> None:
        mapped = twixtools.map_twix(filename)
    
        im_data = mapped[-1]['image']
        
        im_data.flags['remove_os'] = True

        self.kspace = im_data[:]

    def run(self) -> None:

        if self._verbose:
            logging.info(
                f"Performing inference on the input data at {self.input_path}."
            )
            logging.info(f"Saving the output data at {self.output_path}.")

        model = torch.load(self.model_path, map_location=self.device)
        model.eval()

        times = []

        for file in tqdm(glob(f"{self.input_path}/*.dat"), desc="Inference (Batch)"):
            start_time = time.time()

            self.load_twix(file)
            self.kspace.to(self.device)
            self.kspace = self.kspace[None, None, ...]

            with torch.no_grad():
                pred = model(self.kspace)

            pred = pred.detach()

            end_time = time.time()

            times.append(round((end_time - start_time), 3))

            if self._verbose:
                logging.info(f"Time for volume: {round((end_time - start_time), 3)} seconds.")

        logging.info(times)
