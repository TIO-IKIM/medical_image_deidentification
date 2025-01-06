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
from torchvision.transforms.functional import InterpolationMode

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
            self.model_path = "model/mednext_deface.pth"
            self.stem = "deface"  # stem for output file
        elif skullstrip:
            self.model_path = "model/mednext_skullstrip.pth"
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

    def run(self) -> None:
        """
        Runs the inference process on the input data and saves the output data.

        This method performs the following steps:
        1. Loads the trained model.
        2. Sets the model to evaluation mode.
        3. Loads the input data using the `get_inference_loader` function.
        4. Iterates over the data and performs inference on each batch.
        5. Processes the predicted output and saves it as a NIfTI image.

        Note: The input data can be in DICOM or NIfTI format.

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

        model = torch.load(self.model_path, map_location=self.device)
        model.eval()

        test_loader = get_inference_loader(self.input_path, batch_size=1)

        times = []

        for data in tqdm(test_loader, desc="Inference (Batch)"):
            start_time = time.time()

            image = data["image"].to(self.device)

            with torch.no_grad():
                pred = model(image.float())

            pred = pred.detach()

            for idx in range(image.shape[0]):
                if data["file_name"][idx].endswith('.nii') or data["file_name"][idx].endswith('.nii.gz'):
                    nifti_img = nib.load(data["file_name"][idx])
                    nifti_img = resample(nifti_img)
                else:
                    nifti_img = dcm2nifti(data["file_name"][idx], transpose=False)

                nifti_img_data = nifti_img.get_fdata()
                size = np.transpose(nifti_img_data, (2, 0, 1)).shape

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
                pred = nifti_img_data * (np.transpose(pred_numpy > 0.5, (1, 2, 0)))
                img = nib.Nifti1Image(pred, nifti_img.affine)
                if data["file_name"][idx].endswith('.nii') or data["file_name"][idx].endswith('.nii.gz'):
                    nib.save(img,
                             f"{self.output_path}/{Path(Path(data['file_name'][idx]).stem).stem}_{self.stem}.nii.gz")
                else:
                    nifti2dcm(img, data["file_name"][idx],
                              f"{self.output_path}")

                end_time = time.time()

                times.append(round((end_time - start_time), 3))

                if self._verbose:
                    logging.info(f"Time for volume: {round((end_time - start_time), 3)} seconds.")

        logging.info(times)
