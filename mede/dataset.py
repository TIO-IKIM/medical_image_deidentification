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
from collections import defaultdict

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
    # Try to get ImageOrientationPatient, otherwise use default orientation
    if hasattr(dicom_first, "ImageOrientationPatient"):
        image_orient1 = np.array(dicom_first.ImageOrientationPatient)[0:3]
        image_orient2 = np.array(dicom_first.ImageOrientationPatient)[3:6]
        image_pos = np.array(dicom_first.ImagePositionPatient)
        last_image_pos = np.array(dicom_last.ImagePositionPatient)
    else:
        logging.warning(
            "ImageOrientationPatient not found in DICOM metadata. Using default orientation."
        )
        # Default to axial orientation if not present
        image_orient1 = np.array([1, 0, 0])
        image_orient2 = np.array([0, 1, 0])
        image_pos = np.array([0, 0, 0])
        last_image_pos = np.array([0, 0, 0])
        

    delta_r = float(dicom_first.PixelSpacing[0])
    delta_c = float(dicom_first.PixelSpacing[1])


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
        
    if is_enhanced_dicom(files[0]):
        nifti, _ = extract_volumes_from_enhanced_dicom(files[0])
        if len(nifti) > 1:
            logging.info(f"Enhanced DICOM with {len(nifti)} volumes found. Returning first volume only.")
        nifti = nifti[0]
        
        ### TODO: Handle multiple volumes properly ###
        
    else:
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
    
    # Enhanced DICOM case
    if os.path.isfile(dcm_dir) and is_enhanced_dicom(dcm_dir):
        # Accept both a single NIfTI or a list of NIfTIs
        if isinstance(nifti_file, nib.Nifti1Image):
            nifti_paths = [out_dir + "_tmp.nii.gz"]
            nib.save(nifti_file, nifti_paths[0])
        else:
            # Assume list of NIfTI images or file paths
            nifti_paths = []
            for i, n in enumerate(nifti_file):
                if isinstance(n, nib.Nifti1Image):
                    path = f"{out_dir}_tmp_{i}.nii.gz"
                    nib.save(n, path)
                    nifti_paths.append(path)
                else:
                    nifti_paths.append(n)
                    
        reconstruct_enhanced_dicom_from_niftis(nifti_paths, dcm_dir, out_dir)

        return

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
        fn = file.lower()
        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
            valid_paths["image_path"].append(file)
        elif fn.endswith(".dcm"):
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

def is_enhanced_dicom(dcm_file: str) -> bool:
    """
    Check if a DICOM file is an Enhanced DICOM format.
    
    Args:
        dcm_file (str): Path to the DICOM file.
        
    Returns:
        bool: True if the file is Enhanced DICOM, False otherwise.
    """
    enhanced_uids = [
        '1.2.840.10008.5.1.4.1.1.4.1',  # Enhanced MR Image Storage
        '1.2.840.10008.5.1.4.1.1.2.1',  # Enhanced CT Image Storage
        '1.2.840.10008.5.1.4.1.1.4.3',  # Enhanced MR Color Image Storage
        '1.2.840.10008.5.1.4.1.1.128',  # Enhanced PET Image Storage
        '1.2.840.10008.5.1.4.1.1.2.2',  # Legacy Converted Enhanced CT
        '1.2.840.10008.5.1.4.1.1.4.4'   # Legacy Converted Enhanced MR
    ]
    
    dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
    
    return hasattr(dcm, "SOPClassUID") and str(dcm.SOPClassUID) in enhanced_uids

def extract_volumes_from_enhanced_dicom(dcm_file: str) -> tuple[list[nib.Nifti1Image], pydicom.Dataset]:
    """
    Extract individual volumes from an Enhanced DICOM file.
    
    Enhanced DICOM files can contain multiple volumes/acquisitions stored as frames.
    This function separates them based on metadata (e.g., different acquisitions,
    different stack IDs, or temporal positions).
    
    Args:
        dcm_file (str): Path to the Enhanced DICOM file.
        
    Returns:
        tuple: (list of NIfTI volumes, original DICOM dataset for metadata)
    """
    dcm = pydicom.dcmread(dcm_file)
    
    if not hasattr(dcm, 'NumberOfFrames'):
        # Single frame, treat as single volume
        volume = dcm.pixel_array
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        affine = create_affine_enhanced(dcm, frame_indices=[0])
        nifti = nib.Nifti1Image(volume.transpose(2, 1, 0), affine)
        return [resample(nifti)], dcm
    
    num_frames = int(dcm.NumberOfFrames)
    pixel_array = dcm.pixel_array
    
    # Group frames by volume using metadata
    volume_groups = group_frames_by_volume(dcm, num_frames)
    
    volumes = []
    for frame_indices in volume_groups:
        # Extract frames for this volume
        volume_data = pixel_array[frame_indices]
        
        # Create affine matrix for this volume
        affine = create_affine_enhanced(dcm, frame_indices)
        
        # Create NIfTI image
        volume_data_transposed = np.transpose(volume_data, (2, 1, 0))
        nifti = nib.Nifti1Image(volume_data_transposed, affine)
        volumes.append(resample(nifti))
    
    return volumes, dcm

def group_frames_by_volume(dcm: pydicom.Dataset, num_frames: int) -> list[list[int]]:
    """
    Group frames into separate volumes based on DICOM metadata.
    
    This looks at PerFrameFunctionalGroupsSequence to identify which frames
    belong to which volume based on:
    - StackID (if available)
    - InStackPositionNumber
    - TemporalPositionIndex
    - DimensionIndexValues
    
    Args:
        dcm (pydicom.Dataset): The Enhanced DICOM dataset.
        num_frames (int): Total number of frames.
        
    Returns:
        list[list[int]]: List of frame index groups, each representing a volume.
    """
    if not hasattr(dcm, 'PerFrameFunctionalGroupsSequence'):
        # No frame-specific metadata, treat all frames as one volume
        return [list(range(num_frames))]
    
    frame_metadata = []
    
    for frame_idx in range(num_frames):
        per_frame = dcm.PerFrameFunctionalGroupsSequence[frame_idx]

        stack_id = None
        temporal_pos = None
        in_stack_pos = None
        dimension_idx = None
        
        # Check for FrameContentSequence
        if hasattr(per_frame, 'FrameContentSequence') and len(per_frame.FrameContentSequence) > 0:
            frame_content = per_frame.FrameContentSequence[0]
            
            if hasattr(frame_content, 'StackID'):
                stack_id = frame_content.StackID
            if hasattr(frame_content, 'InStackPositionNumber'):
                in_stack_pos = int(frame_content.InStackPositionNumber)
            if hasattr(frame_content, 'TemporalPositionIndex'):
                temporal_pos = int(frame_content.TemporalPositionIndex)
            if hasattr(frame_content, 'DimensionIndexValues'):
                dimension_idx = tuple(frame_content.DimensionIndexValues)
        
        frame_metadata.append({
            'frame_idx': frame_idx,
            'stack_id': stack_id,
            'temporal_pos': temporal_pos if temporal_pos is not None else 1,
            'in_stack_pos': in_stack_pos,
            'dimension_idx': dimension_idx
        })
    
    # Group by stack_id first, then by temporal_pos
    volume_dict = defaultdict(list)
    
    for frame_meta in frame_metadata:
        # Create a key that identifies unique volumes
        # Prioritize stack_id, then temporal_pos
        if frame_meta['stack_id'] is not None:
            key = (frame_meta['stack_id'], frame_meta['temporal_pos'])
        elif frame_meta['dimension_idx'] is not None:
            # For multi-dimensional data, use first dimension index as volume identifier
            key = (frame_meta['dimension_idx'][0] if len(frame_meta['dimension_idx']) > 0 else 0,
                   frame_meta['temporal_pos'])
        else:
            # Fall back to temporal position only
            key = (0, frame_meta['temporal_pos'])
        
        volume_dict[key].append(frame_meta['frame_idx'])
    
    # Sort frames within each volume by in_stack_pos
    volume_groups = []
    for key in sorted(volume_dict.keys()):
        frame_indices = volume_dict[key]
        # Sort by in_stack_pos if available
        if frame_metadata[frame_indices[0]]['in_stack_pos'] is not None:
            frame_indices = sorted(frame_indices, 
                                 key=lambda idx: frame_metadata[idx]['in_stack_pos'])
        volume_groups.append(frame_indices)
    
    return volume_groups

def create_affine_enhanced(dcm: pydicom.Dataset, frame_indices: list[int]) -> np.matrix:
    """
    Create an affine matrix for Enhanced DICOM frames.
    
    Args:
        dcm (pydicom.Dataset): Enhanced DICOM dataset.
        frame_indices (list[int]): List of frame indices for this volume.
        
    Returns:
        np.matrix: The affine matrix.
    """
    first_frame_idx = frame_indices[0]
    last_frame_idx = frame_indices[-1]
    
    # Get orientation from SharedFunctionalGroupsSequence or first frame
    if hasattr(dcm, 'SharedFunctionalGroupsSequence') and len(dcm.SharedFunctionalGroupsSequence) > 0:
        shared = dcm.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, 'PlaneOrientationSequence') and len(shared.PlaneOrientationSequence) > 0:
            image_orient = np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient)
            image_orient1 = image_orient[0:3]
            image_orient2 = image_orient[3:6]
        else:
            image_orient1 = np.array([1, 0, 0])
            image_orient2 = np.array([0, 1, 0])
    else:
        image_orient1 = np.array([1, 0, 0])
        image_orient2 = np.array([0, 1, 0])
    
    # Get position from PerFrameFunctionalGroupsSequence
    per_frame_first = dcm.PerFrameFunctionalGroupsSequence[first_frame_idx]
    per_frame_last = dcm.PerFrameFunctionalGroupsSequence[last_frame_idx]
    
    if hasattr(per_frame_first, 'PlanePositionSequence') and len(per_frame_first.PlanePositionSequence) > 0:
        image_pos = np.array(per_frame_first.PlanePositionSequence[0].ImagePositionPatient)
    else:
        image_pos = np.array([0, 0, 0])
    
    if hasattr(per_frame_last, 'PlanePositionSequence') and len(per_frame_last.PlanePositionSequence) > 0:
        last_image_pos = np.array(per_frame_last.PlanePositionSequence[0].ImagePositionPatient)
    else:
        last_image_pos = image_pos
    
    # Get pixel spacing
    if hasattr(dcm, 'SharedFunctionalGroupsSequence') and len(dcm.SharedFunctionalGroupsSequence) > 0:
        shared = dcm.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, 'PixelMeasuresSequence') and len(shared.PixelMeasuresSequence) > 0:
            pixel_measures = shared.PixelMeasuresSequence[0]
            if hasattr(pixel_measures, 'PixelSpacing'):
                pixel_spacing = pixel_measures.PixelSpacing
                delta_r = float(pixel_spacing[0])
                delta_c = float(pixel_spacing[1])
            else:
                delta_r = delta_c = 1.0
        else:
            delta_r = delta_c = 1.0
    else:
        delta_r = delta_c = 1.0
    
    # Calculate step between slices
    if len(frame_indices) == 1:
        step = np.array([0, 0, -1])
    else:
        step = (image_pos - last_image_pos) / (1 - len(frame_indices))
    
    affine = np.matrix([
        [-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
        [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
        [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
        [0, 0, 0, 1]
    ])
    
    return affine

def reconstruct_enhanced_dicom_from_niftis(nifti_paths: list, original_dcm_path: str, output_path: str):
    """
    Reconstructs an enhanced DICOM file by replacing its pixel data with data from a list of processed NIfTI files.
    This function reads the original enhanced DICOM file, groups its frames by volume, and replaces the pixel data
    for each volume with the corresponding data from the provided NIfTI files. The resulting DICOM is saved to the
    specified output path.
    
    Args:
        nifti_paths (list): List of file paths to 3D NIfTI files, each representing a processed volume.
        original_dcm_path (str): Path to the original enhanced DICOM file to be used as a template.
        output_path (str): Directory where the reconstructed DICOM file will be saved.
        
    Raises:
        ValueError: If the number of NIfTI files does not match the number of volumes in the original DICOM,
                    or if a NIfTI file is not 3D.
    """

    dcm = pydicom.dcmread(original_dcm_path)
    num_frames = int(dcm.NumberOfFrames)
    
    volume_groups = group_frames_by_volume(dcm, num_frames)
    if len(nifti_paths) != len(volume_groups):
        raise ValueError("Number of NIfTIs does not match number of volumes in original DICOM.")
    all_frames = []
    
    for nifti_path, frame_indices in zip(nifti_paths, volume_groups):
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        else:
            raise ValueError("Processed NIfTI must be 3D.")
        
        if data.shape[0] != len(frame_indices):
            if data.shape[0] < len(frame_indices):
                pad = np.zeros((len(frame_indices) - data.shape[0], data.shape[1], data.shape[2]))
                data = np.concatenate([data, pad], axis=0)
            else:
                data = data[:len(frame_indices)]
        for slice_idx in range(len(frame_indices)):
            all_frames.append(data[slice_idx, ...])
    new_pixel_array = np.array(all_frames)
    
    new_pixel_array = new_pixel_array.astype(dcm.pixel_array.dtype)
    dcm.PixelData = new_pixel_array.tobytes()
    
    pydicom.dcmwrite(f"{output_path}/{original_dcm_path.split('/')[-1]}", dcm)
    
    # Delete temporary NIfTI files
    for nifti_path in nifti_paths:
        if os.path.exists(nifti_path):
            os.remove(nifti_path)