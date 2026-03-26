import pydicom
from pydicom.encaps import encapsulate
import cv2
import numpy as np
import os
import easyocr
from glob import glob
import nibabel as nib
from PIL import Image
from pathlib import Path
import logging


class TextRemoval:
    """
    Class for performing text removal on images using EasyOCR (CRAFT Text Detection).

    Attributes:
        output_path (str): Path to save the output images.
        verbose (bool): If True, enables verbose logging.
        reader (easyocr.Reader): The deep learning OCR model initialized in memory.
        interactive (bool): If True, enables interactive refinement of the output images.

    Methods:
        predict: Apply text removal algorithm to an image.
        __call__: Apply text removal to a directory of images.
    """

    def __init__(self, output_path: str = None, verbose: bool = False, langs: list = ['en'], interactive: bool = False) -> None:
        self.output_path = (
            output_path if output_path is not None else "./text_removed_images"
        )
        self.interactive = interactive
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(f"Saving text removed images to {self.output_path}")

        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize the EasyOCR reader once to keep the model in memory.
        # It will automatically use a GPU if CUDA is available.
        self.reader = easyocr.Reader(langs, gpu=True)
    
    def predict(self, img: np.array, img_orig: np.array = None) -> np.array:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        target_img = img_orig if img_orig is not None else img.copy()
        
        # --- PASS 1: Normal Scale ---
        results_normal = self.reader.readtext(img)
        
        # --- PASS 2: Downscaled for detection of big characters / numbers ---
        results_shrunk = self.reader.readtext(img, mag_ratio=0.1, text_threshold=0.5)
        
        # Combine results
        all_results = results_normal + results_shrunk

        # Draw rectangles
        for bbox, text, prob in all_results:
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]

            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)

            padding = 3
            target_img = cv2.rectangle(
                target_img,
                (max(0, left - padding), max(0, top - padding)),
                (right + padding, bottom + padding),
                (255, 255, 255),
                -1,
            )

        return target_img

    def refine_image(self, img: np.array, window_name: str = "Interactive Refinement") -> np.array:
        """
        Opens an interactive OpenCV window to let the user manually draw white rectangles 
        over remaining text/artifacts.

        Args:
            img (np.array): The image to refine.
            window_name (str): The name of the OpenCV window.

        Returns:
            np.array: The manually refined image.
        """
        # Create copies so we can reset if the user makes a mistake
        original_clone = img.copy()
        display_img = img.copy()
        
        # State variables for mouse tracking
        drawing = False
        ix, iy = -1, -1

        def draw_rectangle(event, x, y, flags, param):
            nonlocal ix, iy, drawing, img, display_img

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    # Draw on a temporary copy so we see the box expanding as we drag
                    display_img = img.copy()
                    cv2.rectangle(display_img, (ix, iy), (x, y), (255, 255, 255), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                # Commit the rectangle to the actual image
                cv2.rectangle(img, (ix, iy), (x, y), (255, 255, 255), -1)
                display_img = img.copy()

        # Set up OpenCV window and attach mouse listener
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # WINDOW_NORMAL allows resizing
        cv2.setMouseCallback(window_name, draw_rectangle)

        print(f"\n--- Refinement Mode: {window_name} ---")
        print("1. Click and drag to draw white boxes over artifacts.")
        print("2. Press 'r' to RESET the image if you make a mistake.")
        print("3. Press 'Enter' or 'Space' to SAVE and continue to the next image.")

        while True:
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF

            # Press Enter (13) or Space (32) to confirm and exit
            if key in [13, 32]: 
                break
            # Press 'r' to reset
            elif key == ord('r'):
                print("Image reset.")
                img = original_clone.copy()
                display_img = original_clone.copy()

        cv2.destroyWindow(window_name)
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
            # Skip directories
            if os.path.isdir(filepath):
                continue
                
            img_orig = None
            file_ending = filepath.split(".")[-1].lower()
            
            match file_ending:
                case "png" | "jpg":
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    img_was_greyscale = len(img.shape) == 2 or (
                        len(img.shape) == 3 and img.shape[2] == 1
                    )
                    base_fn = filepath[:-4]
                case "jpeg":
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    img_was_greyscale = len(img.shape) == 2 or (
                        len(img.shape) == 3 and img.shape[2] == 1
                    )
                    base_fn = filepath[:-5]
                case "dcm":
                    dcm = pydicom.dcmread(filepath, force=True)
                    img_orig = pydicom.pixel_array(dcm)
                    img_was_greyscale = len(img_orig.shape) == 2 or (
                        len(img_orig.shape) == 3 and img_orig.shape[2] == 1
                    )
                    img = np.array(
                        Image.fromarray(img_orig).convert(
                            "L" if img_was_greyscale else "RGB"
                        )
                    )
                    base_fn = filepath[:-4]
                case "nii":
                    nifti = nib.load(filepath)
                    nii_data = nifti.get_fdata().squeeze()
                    img_was_greyscale = len(nii_data.shape) == 2 or (
                        len(nii_data.shape) == 3 and nii_data.shape[2] == 1
                    )
                    img = np.array(
                        Image.fromarray(nii_data).convert(
                            "L" if img_was_greyscale else "RGB"
                        )
                    )
                    base_fn = filepath[:-4]
                case "gz":
                    nifti = nib.load(filepath)
                    nii_data = nifti.get_fdata().squeeze()
                    img_was_greyscale = len(nii_data.shape) == 2 or (
                        len(nii_data.shape) == 3 and nii_data.shape[2] == 1
                    )
                    img = np.array(
                        Image.fromarray(nii_data).convert(
                            "L" if img_was_greyscale else "RGB"
                        )
                    )
                    base_fn = filepath[:-7]
                case _:
                    continue

            img = self.predict(
                img=img, img_orig=img_orig if "img_orig" in locals() and img_orig is not None else None
            )

            # Optional manual refinement step for any remaining text/artifacts
            if self.interactive:
                img = self.refine_image(img, window_name=f"Refining: {Path(base_fn).name}")

            # Convert back to greyscale if the input was greyscale
            if img_was_greyscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            _output_path = os.path.join(
                self.output_path, f"{Path(base_fn).name}_text_removed"
            )

            match file_ending:
                case "png":
                    cv2.imwrite(f"{_output_path}.png", img)
                case "jpg" | "jpeg":
                    cv2.imwrite(f"{_output_path}.jpg", img)
                case "dcm":
                    # Check if the pixel data is compressed
                    if (
                        hasattr(dcm.file_meta, "TransferSyntaxUID")
                        and dcm.file_meta.TransferSyntaxUID.is_compressed
                    ):
                        # Re-encapsulate the pixel data if compression is required
                        dcm.PixelData = encapsulate([img.tobytes()])
                        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                    else:
                        dcm.PixelData = img.tobytes()
                    dcm.save_as(f"{_output_path}.dcm")
                case "nii":
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f"{_output_path}.nii")
                case "gz":
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f"{_output_path}.nii.gz")