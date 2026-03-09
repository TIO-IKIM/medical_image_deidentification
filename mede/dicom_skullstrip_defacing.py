# -*- coding: utf-8 -*-
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from mede.dataset import get_inference_loader, resample, dcm2nifti, nifti2dcm
from mede.model import Mednext
import numpy as np
from torchvision.transforms import v2
import nibabel as nib
import time
from torchvision.transforms.functional import InterpolationMode
import importlib.resources

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
        if deface:
            self.model_path = f"{importlib.resources.files('mede')}/models/mednext_deface.pth"
            self.stem = "deface"  # stem for output file
        elif skullstrip:
            self.model_path = f"{importlib.resources.files('mede')}/models/mednext_skullstrip.pth"
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
    
    @staticmethod
    def _deidentify_header(nii_image, keys: list = ["descrip", "intent_name"]) -> nib.Nifti1Image:
        """
        De-identify the header of a NIfTI image by clearing specified metadata fields.
        Parameters:
        nii_image (nib.Nifti1Image): The NIfTI image whose header is to be de-identified.
        keys (list): A list of header keys to be cleared. Default is ["descrip", "intent_name"].
        Returns:
        nib.Nifti1Image: The NIfTI image with the specified header fields cleared.
        """
        for key in keys:
            if key in nii_image.header:
                nii_image.header[key] = b""
        return nii_image
        
    def run(self) -> None:
        """
        Runs the inference process on the input data and saves the output data.

        This method performs the following steps:
        1. Loads the trained model.
        2. Sets the model to evaluation mode.
        3. Loads the input data using the `get_inference_loader` function.
        4. Iterates over the data and performs inference on each batch.
        5. Processes the predicted output and saves it as a NIfTI image.

        Note: The input data can be in DICOM, NIfTI, or NumPy (.npy) volume format.

        Raises:
            None

        Returns:
            None
        """
        if self._verbose:
            logging.info(
                f"Performing inference on the input data at {self.input_path}."
            )
            logging.info(f"Saving the output data at {self.output_path}.")

        model = Mednext().to(self.device)
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        model.eval()

        test_loader = get_inference_loader(self.input_path, batch_size=1)

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        times = []

        for data in tqdm(test_loader, desc="Inference (Batch)"):
            start_time = time.time()

            image = data["image"].to(self.device)

            with torch.no_grad():
                pred = model(image.float())

            pred = pred.detach()

            for idx in range(image.shape[0]):
                file_path = data["file_name"][idx]
                file_lower = file_path.lower()
                is_nifti = file_lower.endswith('.nii') or file_lower.endswith('.nii.gz')
                is_npy = file_lower.endswith('.npy')

                if is_nifti:
                    nifti_img = nib.load(file_path)
                    nifti_img = resample(nifti_img)
                    input_volume = nifti_img.get_fdata()
                    affine = nifti_img.affine
                    header = nifti_img.header
                elif is_npy:
                    input_volume = np.load(file_path)
                    if input_volume.ndim == 4 and input_volume.shape[0] == 1:
                        input_volume = input_volume[0]
                    if input_volume.ndim != 3:
                        raise ValueError(
                            f"Unsupported numpy array shape {input_volume.shape} in {file_path}. Expected 3D volume."
                        )
                    affine = np.eye(4)
                    header = None
                else:
                    nifti_img = dcm2nifti(file_path, transpose=False)
                    input_volume = nifti_img.get_fdata()
                    affine = nifti_img.affine
                    header = nifti_img.header

                size = np.transpose(input_volume, (2, 0, 1)).shape

                if pred.ndim == 5:
                    pred_numpy = torch.nn.Upsample(size=size)(
                        pred[idx].unsqueeze(dim=0)).cpu().numpy().squeeze().squeeze()
                else:
                    pred_numpy = (
                        v2.functional.resize(
                            pred[idx], size[-2:], antialias=True, interpolation=InterpolationMode.BICUBIC
                        )
                        .numpy()
                    )
                if pred_numpy.ndim == 2:
                    pred_numpy = pred_numpy[None, ...]
                pred_volume = input_volume * (np.transpose(pred_numpy > 0.5, (1, 2, 0)))

                if is_npy:
                    out_path = Path(self.output_path) / f"{Path(Path(file_path).stem).stem}_{self.stem}.npy"
                    np.save(out_path, pred_volume)
                else:
                    img = nib.Nifti1Image(pred_volume, affine=affine, header=header)
                    img = self._deidentify_header(img)

                    if is_nifti:
                        nib.save(
                            img,
                            f"{self.output_path}/{Path(Path(file_path).stem).stem}_{self.stem}.nii.gz",
                        )
                    else:
                        nifti2dcm(img, file_path, f"{self.output_path}")

                end_time = time.time()

                times.append(round((end_time - start_time), 3))

                if self._verbose:
                    logging.info(f"Time for volume: {round((end_time - start_time), 3)} seconds.")
