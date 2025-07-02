#!/usr/bin/env python

# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import re
import struct
import random
from pathlib import Path
import logging
from typing import IO
from tqdm import tqdm
from glob import glob
import pandas as pd
import os
from datetime import datetime

logging.basicConfig(
    encoding="utf-8", level=logging.DEBUG, format="%(levelname)s - %(message)s"
)

class TwixAnonymizer:
    def __init__(
        self,
        filename: str,
        save_path: str,
        csv_path: str = None,
        meta_only: bool = False,
    ) -> None:
        """
        Anonymizes TWIX files by replacing sensitive information with placeholders.

        Args:
            filename (str): The path to the TWIX file to be anonymized.
            save_path (str): The directory where the anonymized TWIX file will be saved.
            csv_path (str, optional): The path to the CSV file where the matches will be written. Defaults to None.
            meta_only (bool, optional): If True, only save the metadata, but do not write anonymized file. Defaults to False.

        Attributes:
            filename (str): The path to the TWIX file to be anonymized.
            save_path (str): The directory where the anonymized TWIX file will be saved.
            csv_path (str): The path to the CSV file where the matches will be written.
            matches (dict): A dictionary of the matched values during anonymization.
            meta_only (bool, optional): If True, only save the metadata, but do not write anonymized file. Defaults to False.

        Methods:
            read_and_anonymize: Reads a TWIX file, determines its type, and performs anonymization based on the file type.
            write_csv: Writes a dictionary of matches to a CSV file.
            anonymize_twix_header: Anonymizes the header string of a TWIX file by replacing sensitive information with placeholders.
            anonymize_twix_vd: Anonymizes a TWIX VD file.
            anonymize_twix_vb: Anonymizes a TWIX VB file.
        """
        self.filename = filename
        self.save_path = save_path
        self.csv_path = csv_path
        self.meta_only = meta_only

    def read_and_anonymize(self) -> None:
        """
        Reads the file, determines its type, and performs anonymization based on the file type.

        This method reads the file specified by `self.filename` and determines its type by checking the first two uints in the header.
        Based on the file type, it performs anonymization using the appropriate method (`anonymize_twix_vd` or `anonymize_twix_vb`).
        The anonymized data is then written to a new file in the `self.save_path` directory.
        If `self.meta_only` is True, only the metadata is anonymized and the anonymized data file is deleted.

        Returns:
            None
        """
        with open(self.filename, "rb") as fin:
            # we can tell the type of file from the first two uints in the header
            first_uint, second_uint = struct.unpack("II", fin.read(8))

            # reset the file pointer before giving to specific function
            fin.seek(0)

            with open(
                Path(self.save_path, f"{str(random.randint(0, 10000))}.dat"), "wb"
            ) as fout:
                if first_uint == 0 and second_uint <= 64:
                    self.filename, self.matches = self.anonymize_twix_vd(
                        fin, fout, meta_only=self.meta_only
                    )
                else:
                    self.filename, self.matches = self.anonymize_twix_vb(
                        fin, fout, meta_only=self.meta_only
                    )

                self.write_csv()

            fout.close()

            if self.meta_only:
                os.remove(fout.name)

    def write_csv(self) -> None:
        """
        Write the matches to a CSV file.

        This method takes the matches stored in the `self.matches` attribute and writes them to a CSV file.
        If a `csv_path` is provided, the matches are appended to an existing CSV file or a new file is created.
        If no `csv_path` is provided, the matches are written to a new CSV file with the same name as the input file.

        Returns:
            None
        """

        anonymized_id = Path(self.filename).stem
        self.matches = {"anonymized_id": anonymized_id, **self.matches}
        if self.csv_path:
            self.filename = self.csv_path
            if Path(self.filename).is_file():
                df = pd.DataFrame(self.matches, index=[0])
                df_orig = pd.read_csv(self.csv_path, index_col=0)
                df = pd.concat([df_orig, df], ignore_index=True)
                df.to_csv(self.filename, mode="w")
            else:
                df = pd.DataFrame(self.matches, index=[0])
                df.to_csv(self.filename, mode="w")
        else:
            df = pd.DataFrame(self.matches, index=[0])
            df.to_csv(self.filename, mode="w")

    @staticmethod
    def _get_date(date_str: str) -> str:
        """
        Converts a date string in the format "%d%m%y" to the format "%y%m%d".

        Args:
            date_str (str): The date string to be converted.

        Returns:
            str: The converted date string in the format "%Y-%m-%d".
        """
        # Parse the string into a datetime object
        date_obj = datetime.strptime(date_str, "%y%m%d")

        # Format the datetime object into the desired format
        formatted_date = date_obj.strftime("%Y-%m-%d")

        return formatted_date

    @staticmethod
    def anonymize_twix_header(header_string: str) -> str | dict:
        """
        Anonymizes the header string of a TWIX file by replacing sensitive information with placeholders.

        Args:
            header_string (str): The header string of the TWIX file.

        Returns:
            tuple: A tuple containing the anonymized header string and a dictionary of the matched values.

        Credit:
            This method was partially adapted from the original implementation by the authors of https://github.com/openmrslab/suspect/blob/master/suspect/io/twix.py
        """
        number_buffer = {
            "Patient_id": r"(<ParamString.\"PatientID\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "Device_serial": r"(<ParamString.\"DeviceSerialNumber\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "Exam_memory_uid": r"(<ParamString.\"ExamMemoryUID\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "PatientLOID": r"(<ParamString.\"PatientLOID\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "StudyLOID": r"(<ParamString.\"StudyLOID\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "SeriesLOID": r"(<ParamString.\"SeriesLOID\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "Study": r"(<ParamString.\"Study\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "FrameOfReference": r"(<ParamString.\"FrameOfReference\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "Patient": r"(<ParamString.\"Patient\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "MeasUID": r"(<ParamString.\"MeasUID\">\s*\{\s*\")(.+)(\"\s*\}\n)",
        }
        x_buffer = {
            "Patient_name": r"(<ParamString.\"t?Patients?Name\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*\")(.+)(\"\s*\}\n)",
            "InstitutionAddress": r"(<ParamString.\"InstitutionAddress\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*\")(.+)(\"\s*\}\n)",
            "InstitutionName": r"(<ParamString.\"InstitutionName\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*\")(.+)(\"\s*\}\n)",
        }
        zero_buffer = {
            "Patient_gender": r"(<ParamLong.\"l?PatientSex\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*)(\d+)(\s*\}\n)",
            "Patient_age": r"(<ParamDouble.\"flPatientAge\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "Patient_weight": r"(<ParamDouble.\"flUsedPatientWeight\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "Patient_height": r"(<ParamDouble.\"flPatientHeight\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*<Unit> \"\[mm\]\"\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "Patient_birthday": r"(<ParamString.\"PatientBirthDay\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*\")(\d{8})(\"\s*\}\n)",
            "ulVersion": r"(<ParamLong.\"ulVersion\">\s*\{(\s*<Visible>\s*\"true\"\s*)?\s*)(\d+)(\s*\}\n)",
        }
        meta_buffer = {
            "tBodyPartExamined": r"(<ParamString.\"tBodyPartExamined\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "Sequence": r"(<ParamString.\"SequenceDescription\">\s*\{\s*\")(.+)(\"\s*\}\n)",
            "TurboFactor": r"(<ParamLong.\"TurboFactor\">\s*\{\s*)(\d+)(\s*\}\n)",
            "ReadoutOversamplingFactor": r"(<ParamDouble.\"ReadoutOversamplingFactor\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "NSlc": r"(<ParamLong.\"NSlc\">\s*\{\s*)(\d+)(\s*\}\n)",
            "PhaseEncodingLines": r"(<ParamLong.\"PhaseEncodingLines\">\s*\{\s*)(\d+)(\s*\}\n)",
            "ReadFoV": r"(<ParamDouble.\"ReadFoV\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "PhaseFoV": r"(<ParamDouble.\"PhaseFoV\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "PhaseResolution": r"(<ParamDouble.\"PhaseResolution\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "TR": r"(<ParamDouble.\"TR\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "TI": r"(<ParamDouble.\"TI\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "flMagneticFieldStrength": r"(<ParamDouble.\"flMagneticFieldStrength\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)",
            "PatientPosition": r"(<ParamString.\"PatientPosition\">\s*\{\s*\")(.+)(\"\s*\}\n)",
        }

        matches = {}

        frame_of_reference = re.search(
            r"(<ParamString.\"FrameOfReference\">  { )(\".+\")(  }\n)", header_string
        ).group(2)
        exam_date_time = frame_of_reference.split(".")[10]
        exam_date = exam_date_time[2:8]
        matches["Exam_date"] = TwixAnonymizer._get_date(exam_date)

        for key, buffer in number_buffer.items():
            match = re.search(buffer, header_string)
            if match:
                matches[key] = match.group(2)
                header_string = re.sub(
                    buffer,
                    lambda match: "".join(
                        (match.group(1), ("0" * (len(match.group(2)))), match.group(3))
                    ),
                    header_string,
                )

        for key, buffer in zero_buffer.items():
            match = re.search(buffer, header_string)
            if match:
                matches[key] = match.group(3)
                header_string = re.sub(
                    buffer,
                    lambda match: "".join(
                        (
                            match.group(1),
                            re.sub(r"\d", "0", match.group(3)),
                            match.group(4),
                        )
                    ),
                    header_string,
                )

        for key, buffer in x_buffer.items():
            match = re.search(buffer, header_string)
            matches[key] = match.group(3)
            header_string = re.sub(
                buffer,
                lambda match: "".join(
                    (
                        match.group(1),
                        ("x" * (len(match.group(3)))),
                        match.group(4),
                    )
                ),
                header_string,
            )

        # Do not anonymize these buffers, but save them
        for key, buffer in meta_buffer.items():
            match = re.search(buffer, header_string)
            if match:
                matches[key] = match.group(2)

        header_string = re.sub(
            r"\"[\d\.]*{0}[\d\.]*\"".format(exam_date),
            lambda match: re.sub(r"\w", "x", match.group()),
            header_string,
        )

        return header_string, matches

    @staticmethod
    def anonymize_twix_vd(fin: IO, fout: IO, meta_only: bool = False) -> str | dict:
        """
        Anonymizes a TWIX VD file.

        Args:
            fin (file): The input file object.
            fout (file): The output file object.
            meta_only (bool, optional): If True, only save the metadata, but do not write anonymized file. Defaults to False.

        Returns:
            Union[str, dict]: The name of the output file and a dictionary of matches found during anonymization.

        Credit:
            This method was adapted from the original implementation by the authors of https://github.com/openmrslab/suspect/blob/master/suspect/io/twix.py
        """
        twix_id, num_measurements = struct.unpack("II", fin.read(8))

        if not meta_only:
            fout.write(struct.pack("II", twix_id, num_measurements))

        for i in range(num_measurements):
            fin.seek(8 + 152 * i)
            meas_id, file_id, offset, length, patient_name, protocol_name = (
                struct.unpack("IIQQ64s64s", fin.read(152))
            )
            anon_patient_name = ("x" * 64).encode("latin-1")

            fin.seek(offset)
            # read the header and anonymize it
            header_size = struct.unpack("I", fin.read(4))[0]
            header = fin.read(header_size - 4)
            header_string = header[:-24].decode("latin-1")

            anonymized_header, matches = TwixAnonymizer.anonymize_twix_header(
                header_string=header_string
            )

            if not meta_only:
                fout.seek(8 + 152 * i)
                fout.write(
                    struct.pack(
                        "IIQQ64s64s",
                        meas_id,
                        file_id,
                        offset,
                        length,
                        anon_patient_name,
                        protocol_name,
                    )
                )

                fout.seek(offset)
                fout.write(struct.pack("I", header_size))
                fout.write(anonymized_header.encode("latin1"))
                fout.write(header[-24:])
                fout.write(fin.read(length - header_size))

        return fout.name, matches

    @staticmethod
    def anonymize_twix_vb(fin: IO, fout: IO, meta_only: bool = False) -> str | dict:
        """
        Anonymizes a TWIX VB file.

        Args:
            fin (file): The input file object.
            fout (file): The output file object.
            meta_only (bool, optional): If True, only save the metadata, but do not write anonymized file. Defaults to False.

        Returns:
            Union[str, dict]: The name of the output file and a dictionary of matches found during anonymization.

        Credit:
            This method was adapted from the original implementation by the authors of https://github.com/openmrslab/suspect/blob/master/suspect/io/twix.py
        """

        # first four bytes are the size of the header
        header_size = struct.unpack("I", fin.read(4))[0]

        # read the rest of the header minus the four bytes we already read
        header = fin.read(header_size - 4)
        # last 24 bytes of the header contain non-strings
        header_string = header[:-24].decode("latin-1")

        anonymized_header, matches = TwixAnonymizer.anonymize_twix_header(
            header_string=header_string
        )

        if not meta_only:
            fout.write(struct.pack("I", header_size))
            fout.write(anonymized_header.encode("latin-1"))
            fout.write(header[-24:])
            fout.write(fin.read())

        return fout.name, matches


def anonymize_twix(input_path: str, save_path: str, meta_only: bool = False):
    """
    Anonymizes TWIX files located at the given input path and saves the anonymized files at the specified save path.

    Args:
        input_path (str): The path to the TWIX file or directory containing TWIX files to be anonymized.
        save_path (str): The path to save the anonymized files.
        meta_only (bool, optional): If True, only save the metadata, but do not write anonymized file. Defaults to False.

    Raises:
        FileNotFoundError: If the input_path does not exist.

    """
    assert Path(input_path).exists(), f"{input_path} does not exist."

    os.makedirs(save_path, exist_ok=True)

    if Path(input_path).is_dir():
        files = [path for path in glob(f"{input_path}/*.dat")]
        folder_len = len(files)
        if meta_only:
            logging.info(f"Only saving metadata! Not writing anonymized files.")
        else:
            logging.info(f"Will save anonymized files in {save_path}.")
        logging.info(f"Anonymizing all files in {input_path}.")
        logging.info(f"Total of {folder_len} files.")

        csv_path = Path(save_path, f"{Path(input_path).name}.csv")

        for filename in tqdm(
            files, desc="Anonymizing files", total=folder_len, unit="files"
        ):
            anonymizer = TwixAnonymizer(filename, save_path, csv_path, meta_only)
            anonymizer.read_and_anonymize()

    else:
        logging.info(f"Anonymizing {input_path}.")
        csv_path = Path(save_path, f"{Path(input_path).stem}.csv")
        if meta_only:
            logging.info(f"Only saving metadata! Not writing anonymized files.")
        anonymizer = TwixAnonymizer(input_path, save_path, csv_path, meta_only)
        anonymizer.read_and_anonymize()