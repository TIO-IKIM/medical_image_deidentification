# -*- coding: utf-8 -*-
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import time
import twixtools
from glob import glob
from utils.fourier import ifft, fft
from sigpy import mri as mr
import sigpy as sp
import nibabel as nib

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
        self.resize = T.Resize((256, 256))

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
        
    def espirit_combine(self, kspace, mps):    
        """
        Combines multi-channel k-space using
        explicit spatial coil-sensitivity maps.

            Parameters:
                kspace (ndarray): raw kspace with channels as first dimension
                mps (ndarray): coil sensitivity maps

            Returns
                combined kspace (nd-array)
        """
        
        # do FFT (k-space --> images space)
        img = sp.ifft(kspace, axes=(1, 2))

        # define coil sensitivity operator
        C = sp.linop.Multiply(img.shape[1:], mps)                    
            
        # coil-combine images and compute combined k-space
        finalImage = C.H * img

        # do FFT (image --> k-space)    
        finalKspace = sp.fft(finalImage, axes=(0, 1))

        return finalKspace
        
    def load_twix(self, filename: str) -> None:
        """
        Reads Siemens Raw Data (TWIX) files.

            Parameters:
                fname (str): File Name

            Returns
                kspace (nd-array): raw k-space data

        """

        # map the twix data to twix_array objects
        mapped = twixtools.map_twix(filename)
        
        #  step 1: load image data
        im_data = mapped[-1]['image']
        
        # the twix_array object makes it easy to remove the 2x oversampling in read direction
        im_data.flags['remove_os'] = True

        # read the data (array-slicing is also supported)
        self.kspace = im_data[:].squeeze()
        
        self.kspace = np.swapaxes(self.kspace, 1, 2)

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
            try:
                self.load_twix(file)
            except:
                logging.error(f"Could not load file: {file}")
                continue
            
            pred_volume = []
            for slc in range(self.kspace.shape[0]):
                mps = mr.app.EspiritCalib(self.kspace[slc], 32,  show_pbar=False).run()
                kspace_slc = self.espirit_combine(self.kspace[slc], mps)
                kspace_slc = torch.tensor(kspace_slc[None, None, ...]).to(self.device)
                kspace_slc_ifft = ifft(kspace_slc)
                kspace_slc = fft(self.resize(kspace_slc_ifft.real) + 1j*self.resize(kspace_slc_ifft.imag))
                kspace_slc /= kspace_slc.std()

                with torch.no_grad():
                    pred = model(kspace_slc)

                pred = pred.detach()
                pred_volume.append(pred.cpu().numpy())
            
            pred_volume = np.array(pred_volume)

            img = nib.Nifti1Image(pred_volume, np.eye(4))
            nib.save(img, f"{self.output_path}/{Path(file).stem}.nii.gz")

            end_time = time.time()

            times.append(round((end_time - start_time), 3))

            if self._verbose:
                logging.info(f"Time for volume: {round((end_time - start_time), 3)} seconds.")

        logging.info(times)
