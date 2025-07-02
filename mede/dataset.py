# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import random
import logging
from glob import glob
import os
from pathlib import Path
import numpy as np
import pydicom
import nibabel as nib
import torchio as tio

INPUT_SIZE = [64, 224, 224]
PREPROCESSING_P = 0.5


def resample(nifti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Resamples a NIfTI image to the target orientation.

    Args:
        nifti_img (nib.Nifti1Image): The input NIfTI image to be resampled.

    Returns:
        nib.Nifti1Image: The resampled NIfTI image.

    """
    orig_orientation = nib.orientations.io_orientation(nifti_img.affine)
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))

    transform = nib.orientations.ornt_transform(orig_orientation, target_orientation)

    return nifti_img.as_reoriented(transform)


def prepare_slices(fp: str) -> np.array:
    """
    Reads a DICOM file and returns the pixel array.

    Parameters:
        fp (str): The file path of the DICOM file.

    Returns:
        np.array: The pixel array of the DICOM file.
    """
    return pydicom.dcmread(fp).pixel_array


def order_slices(fp: str) -> int:
    """
    Reads a DICOM file and returns the instance number.

    Parameters:
        fp (str): The file path of the DICOM file.

    Returns:
        int: The instance number of the DICOM file.
    """
    return pydicom.dcmread(fp, stop_before_pixels=True).InstanceNumber


def create_affine(sorted_dicoms: list) -> np.matrix:
    """
    Create an affine matrix based on the DICOM metadata.

    Args:
        sorted_dicoms (list): A list of sorted DICOM file paths.

    Returns:
        numpy.matrix: The affine matrix.

    Adapted from https://dicom2nifti.readthedocs.io/en/latest/_modules/dicom2nifti/common.html#create_affine.
    """

    dicom_first = pydicom.dcmread(sorted_dicoms[0])
    dicom_last = pydicom.dcmread(sorted_dicoms[-1])
    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = np.array(dicom_first.ImageOrientationPatient)[0:3]
    image_orient2 = np.array(dicom_first.ImageOrientationPatient)[3:6]

    delta_r = float(dicom_first.PixelSpacing[0])
    delta_c = float(dicom_first.PixelSpacing[1])

    image_pos = np.array(dicom_first.ImagePositionPatient)

    last_image_pos = np.array(dicom_last.ImagePositionPatient)

    if len(sorted_dicoms) == 1:
        # Single slice
        step = [0, 0, -1]
    else:
        step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))

    affine = np.matrix(
        [
            [
                -image_orient1[0] * delta_c,
                -image_orient2[0] * delta_r,
                -step[0],
                -image_pos[0],
            ],
            [
                -image_orient1[1] * delta_c,
                -image_orient2[1] * delta_r,
                -step[1],
                -image_pos[1],
            ],
            [
                image_orient1[2] * delta_c,
                image_orient2[2] * delta_r,
                step[2],
                image_pos[2],
            ],
            [0, 0, 0, 1],
        ]
    )

    return affine


def dcm2nifti(dir_path: str, transpose: bool = False) -> nib.Nifti1Image:
    """
    Convert DICOM files in a directory to a NIfTI image.

    Args:
        dir_path (str): The path to the directory containing the DICOM files.
        transpose (bool, optional): Whether to transpose the resulting NIfTI image. Defaults to False.

    Returns:
        nib.Nifti1Image: The converted NIfTI image.
    """

    if os.path.isdir(dir_path):
        # sort files by instance number
        files = sorted(
            glob(os.path.join(dir_path, "**", "*.dcm"), recursive=True),
            key=order_slices,
        )
    else:
        files = [dir_path]

    affine = create_affine(files)

    slices = [prepare_slices(f) for f in files]
    volume = np.array(slices)
    volume = np.transpose(volume, (2, 1, 0))

    nifti = nib.Nifti1Image(volume, affine)
    nifti = resample(nifti)

    if transpose:
        nifti = np.transpose(nifti.get_fdata().copy(), (2, 0, 1))

    return nifti


def nifti2dcm(nifti_file: nib.Nifti1Image, dcm_dir: str, out_dir: str) -> None:
    """
    Convert a NIfTI file to a series of DICOM files.

    Args:
        nifti_file (nibabel.Nifti1Image): The NIfTI file to convert.
        dcm_dir (str): The directory containing the DICOM files used for reference.
        out_dir (str): The output directory to save the converted DICOM files.

    Returns:
        None
    """

    if os.path.isdir(dcm_dir):
        files = sorted(
            glob(os.path.join(dcm_dir, "**", "*.dcm"), recursive=True), key=order_slices
        )
    else:
        files = [dcm_dir]
    target_affine = create_affine(files)

    orig_orientation = nib.orientations.io_orientation(nifti_file.affine)
    target_orientation = nib.orientations.io_orientation(target_affine)
    transform = nib.orientations.ornt_transform(orig_orientation, target_orientation)

    nifti_file = nifti_file.as_reoriented(transform)

    nifti_array = nifti_file.get_fdata()
    nifti_array = np.transpose(nifti_array, (2, 1, 0))
    number_slices = nifti_array.shape[0]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for slice_ in range(number_slices):
        dcm = pydicom.dcmread(files[slice_], stop_before_pixels=False)
        dcm.PixelData = (nifti_array[slice_, ...]).astype(np.uint16).tobytes()
        pydicom.dcmwrite(
            filename=os.path.join(out_dir, f"slice{slice_}.dcm"),
            dataset=dcm,
        )


class SegmentationDataset(Dataset):
    """
    A PyTorch dataset for segmentation tasks.

    Args:
        path_list (dict): A list of paths to the image and mask files.
        train (bool): A flag indicating whether the dataset is for training or not.

    Attributes:
        path_list (dict): A list of paths to the image and mask files.
        train (bool): A flag indicating whether the dataset is for training or not.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data at the given index.

    Static Methods:
        normalize(data): Normalizes the input tensor.

    """

    def __init__(self, path_list: dict, train: bool) -> None:
        super().__init__()
        self.path_list = path_list
        self.train = train
        self.up = torch.nn.Upsample(size=(INPUT_SIZE))
        self.transforms_dict = {
            tio.RandomAffine(): 0.75,
            tio.RandomElasticDeformation(): 0.25,
        }

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the data at the given index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            dict: A dictionary containing the image and mask data.
        """
        self.transpose = tio.Lambda(lambda x: x.permute(0, 3, 1, 2))

        image = resample(nib.load(self.path_list["image_path"][idx]))
        mask = resample(nib.load(self.path_list["mask_path"][idx]))

        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=image.get_fdata()[None, ...].copy(), affine=image.affine
            ),
            mask=tio.LabelMap(
                tensor=mask.get_fdata()[None, ...].copy(), affine=mask.affine
            ),
        )

        if self.train:
            transform = tio.OneOf(self.transforms_dict)
            subject = transform(subject)
            if random.random() > 0.25:
                bias = tio.transforms.RandomBiasField()
                subject = bias(subject)
            if random.random() > 0.25:
                noise = tio.transforms.RandomNoise()
                subject = noise(subject)
            if random.random() > 0.25:
                gamma = tio.transforms.RandomGamma()
                subject = gamma(subject)
            if random.random() > 0.25:
                spike = tio.transforms.RandomSpike()
                subject = spike(subject)

        subject = self.transpose(subject)

        data_dict = {
            "image": subject["image"].data,
            "mask": subject["mask"].data,
        }

        data_dict["image"] = v2.Lambda(lambda x: self.up(x.unsqueeze(0)).squeeze(0))(
            data_dict["image"]
        )
        data_dict["mask"] = v2.Lambda(lambda x: self.up(x.unsqueeze(0)).squeeze(0))(
            data_dict["mask"]
        )

        data_dict["image"] = v2.Lambda(lambda x: self._normalize(x))(data_dict["image"])

        return data_dict

    @staticmethod
    def _normalize(data: torch.tensor) -> torch.tensor:
        """
        Normalizes the input tensor.

        Args:
            data (torch.tensor): The input tensor to be normalized.

        Returns:
            torch.tensor: The normalized tensor.

        """
        if data.max() > 0:
            data = (data - data.min()) / (data.max() - data.min())

        return data


class InferenceDataset(SegmentationDataset):
    """
    Dataset class for inference.

    Args:
        path_list (dict): A dictionary containing the paths to the images.

    Attributes:
        up (torch.nn.Upsample): Upsampling layer.
    """

    def __init__(self, path_list: dict) -> None:
        super().__init__(path_list, train=False)
        self.up = torch.nn.Upsample(size=(INPUT_SIZE))

    def __len__(self) -> int:
        return len(self.path_list["image_path"])

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the image and file name.
        """

        file_path = self.path_list["image_path"][idx]

        if file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
            nifti_img = nib.load(file_path)
            pixels = resample(nifti_img).get_fdata().copy()
            pixels = np.transpose(pixels, (2, 0, 1))
        else:
            # assert os.path.isdir(file_path), "DICOM must be volume in folder."
            pixels = dcm2nifti(file_path, transpose=True)

        x = torch.from_numpy(pixels).unsqueeze(dim=0)

        data_dict = {
            "image": x,
            "file_name": file_path,
        }

        data_dict["image"] = v2.Lambda(lambda x: self.up(x.unsqueeze(0)).squeeze(0))(
            data_dict["image"]
        )
        data_dict["image"] = v2.Lambda(lambda x: self._normalize(x))(data_dict["image"])

        return data_dict


def get_loaders(
    train_paths: dict | None = None,
    val_paths: dict | None = None,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns the data loaders for training and validation datasets.

    Args:
        train_paths (dict): List of file paths for training data.
        val_paths (dict): List of file paths for validation data.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    dataloader_params = {
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
    }

    train_dataset = SegmentationDataset(train_paths, train=True)
    val_dataset = SegmentationDataset(val_paths, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, **dataloader_params)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, **dataloader_params)

    return train_loader, val_loader


def get_inference_loader(
    data_path: str,
    batch_size: int = 16,
) -> DataLoader:
    """
    Create a data loader for inference.

    Args:
        data_path (str): The path to the data.
        batch_size (int, optional): The batch size. Defaults to 16.

    Returns:
        DataLoader: The data loader for inference.
    """

    dataloader_params = {
        "batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 4,
    }

    potential_paths = {
        "image_path": (
            glob(os.path.join(data_path, "*"), recursive=True)
            if os.path.isdir(data_path)
            else [data_path]
        )
    }

    valid_paths = {"image_path": []}

    for file in potential_paths["image_path"]:
        file = file.lower()
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            valid_paths["image_path"].append(file)
        elif file.endswith(".dcm"):
            valid_paths["image_path"].append(file)
        elif os.path.isdir(file):
            valid_paths["image_path"].append(file)
        else:
            try:
                pydicom.dcmread(file, stop_before_pixels=True)
                valid_paths["image_path"].append(file)
            except Exception:
                logging.warning(
                    f"Wrong file format: {file}\nAccepted file formats are .nii, .nii.gz and .dcm"
                )

    inference_dataset = InferenceDataset(valid_paths)
    inference_loader = DataLoader(inference_dataset, **dataloader_params)

    return inference_loader
