import pydicom
import cv2 as cv
import numpy as np
import os
import pytesseract
import cv2
from glob import glob
import nibabel as nib
from PIL import Image
from pathlib import Path
import logging


class TextRemoval:
    """
    Class for performing text removal on images.

    Attributes:
        output_path (str): Path to save the output images.
        verbose (bool): If True, enables verbose logging.

    Methods:
        predict: Apply text removal algorithm to an image.
        __call__: Apply text removal to a directory of images.
    """

    def __init__(self, output_path: str = None, verbose: bool = False) -> None:
        self.output_path = output_path if output_path is not None else "./text_removed_images"
        logging.info(f"Saving text removed image to {self.output_path}") if verbose else None
        
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def predict(img: np.array, img_orig: np.array = None) -> np.array:
        """
        Apply text removal algorithm (tesseract) to an image.

        Args:
            img (np.array): Input image as a NumPy array.
            img_orig (np.array, optional): Original image to use for reference. Defaults to None.

        Returns:
            np.array: Image with text removed.
        """

        threshold = 100

        # Insert rectangle in middle of image to ignore this part in the first iteration
        height, width = img.shape[:2]
        left = int(width / 4)
        top = int(height / 4)
        right = int(width/ 4)
        bottom = int(height / 4)

        img_covered = cv2.rectangle(
            img.copy(), (left, top), (right, bottom), (255, 255, 255), -1
        )
        boxes = pytesseract.image_to_boxes(
            img_covered, output_type=pytesseract.Output.DICT, nice=1
        )

        for left, bottom, right, top in zip(
            boxes["left"], boxes["bottom"], boxes["right"], boxes["top"]
        ):
            if right - left < threshold:
                img = cv2.rectangle(
                    img_orig if img_orig is not None else img,
                    (left, height - bottom),
                    (right, height - top),
                    (255, 255, 255),
                    -1,
                )

        # Another iteration without the rectangle in the middle of the image
        try:
            boxes = pytesseract.image_to_boxes(
                img, output_type=pytesseract.Output.DICT, nice=1
            )

            for left, bottom, right, top in zip(
                boxes["left"], boxes["bottom"], boxes["right"], boxes["top"]
            ):
                if right - left < threshold:
                    img = cv2.rectangle(
                        img,
                        (left, height - bottom),
                        (right, height - top),
                        (255, 255, 255),
                        -1,
                    )
        except:
            pass

        return img

    def __call__(self, directory: str) -> None:
        """
        Apply text removal to a directory of images.

        Args:
            directory (str): Path to the directory containing the images.

        Returns:
            None
        """

        if os.path.isdir(directory):
            files = glob(os.path.join(directory, "**", "*"), recursive=True)
        else:
            files = [directory]
        for filepath in files:
            file_ending = filepath.split(".")[-1].lower()
            match file_ending:
                # nifti
                case "png" | "jpg":
                    img = cv.imread(filepath, 0)
                    base_fn = filepath[:-4]
                case "jpeg":
                    img = cv.imread(filepath, 0)
                    base_fn = filepath[:-5]
                case "dcm":
                    dcm = pydicom.dcmread(filepath, force=True)
                    img_orig = dcm.pixel_array
                    img = np.array(
                        Image.fromarray(img_orig).convert("RGB")
                    )
                    base_fn = filepath[:-4]
                case "nii":
                    nifti = nib.load(filepath)
                    img = np.array(
                        Image.fromarray(nifti.get_fdata().squeeze()).convert("RGB")
                    )
                    base_fn = filepath[:-4]
                case "gz":
                    nifti = nib.load(filepath)
                    img = np.array(
                        Image.fromarray(nifti.get_fdata().squeeze()).convert("RGB")
                    )
                    base_fn = filepath[:-7]
                case _:
                    raise NotImplementedError(
                        f"File ending {file_ending} not compatible, must be .dcm, .png, .jpg or .jpeg"
                    )
                    
            img = self.predict(img=img, img_orig=img_orig if 'img_orig' in locals() else None)

            self.output_path = os.path.join(
                self.output_path, f"{Path(base_fn).name}_text_removed"
            )

            match file_ending:
                # nifti
                case "png" | "jpg" | "jpeg":
                    cv.imwrite(f"{self.output_path}.png", img)
                case "dcm":
                    dcm.PixelData = img.tobytes()
                    dcm.save_as(f"{self.output_path}.dcm")
                case "nii":
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f"{self.output_path}.nii")
                case "gz":
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f"{self.output_path}.nii.gz")
