import os
import shutil
import csv
import uuid
import re

class Rename:
    """
    Rename files in a directory to UUIDs, preserving slice order for series, and log the mapping in a CSV file.

    Attributes:
        input_path (str): The directory containing the files to be renamed.
        output_path (str): The directory where renamed files will be copied.
        slice_pattern (re.Pattern): A regular expression pattern to detect file names
            with a series prefix and slice number.
        csv_file (str): The path to the CSV file that logs the mapping of old and new file names.

    Methods:
        __call__():
            Invokes the rename_files method to process the files.
        rename_files():
            Renames files in the input directory, copies them to the output directory,
            and logs the renaming in the CSV file.
        generate_new_name(old_name, series_uuid_map=None):
            Generates a new file name based on the old file name. If the file name matches
            the slice pattern, it preserves the slice number and assigns a UUID to the series.
            Otherwise, it assigns a pure UUID as the new name.
            Args:
                old_name (str): The original file name.
                series_uuid_map (dict, optional): A mapping of series keys to UUIDs.
            Returns:
                str: The newly generated file name.
        log_rename(old_name, new_name):
            Logs the mapping of an old file name to a new file name in the CSV file.
            Args:
                old_name (str): The original file name.
                new_name (str): The new file name.
    """
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.slice_pattern = re.compile(r"^(.*?)(\d+)$")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # Create new csv if none in output path
        self.csv_file = os.path.join(self.output_path, 'rename_mapping.csv')
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['old_name', 'new_name'])  # Write header

    def __call__(self):
        self.rename_files()
        
    def rename_files(self) -> None:
        """
        Renames files in the input directory, copies them to the output directory, and logs the renaming in the CSV file.
        """
        files = [
            filename
            for filename in os.listdir(self.input_path)
            if os.path.isfile(os.path.join(self.input_path, filename))
        ]
        files.sort()

        # Reuse one UUID per detected series, preserve original slice number.
        series_uuid_map = {}

        for filename in files:
            old_file_path = os.path.join(self.input_path, filename)
            new_filename = self.generate_new_name(filename, series_uuid_map)
            new_file_path = os.path.join(self.output_path, new_filename)
            shutil.copy2(old_file_path, new_file_path)
            self.log_rename(filename, new_filename)

    def generate_new_name(self, old_name: str, series_uuid_map=None) -> str:
        """
        Generates a new file name based on the old file name. 
        If the file name matches the slice pattern, it preserves the slice number and assigns a UUID to the series. 
        Otherwise, it assigns a pure UUID as the new name.

        Args:
            old_name (str): The original file name.
            series_uuid_map (dict, optional): A mapping of series keys to UUIDs.

        Returns:
            str: The newly generated file name.
        """
        if series_uuid_map is None:
            series_uuid_map = {}

        name_parts = old_name.rsplit('.', 2)
        base_name = name_parts[0]
        extension = ''.join('.' + part for part in name_parts[1:]) if len(name_parts) > 1 else ''
        match = self.slice_pattern.match(base_name)

        if match:
            series_prefix, slice_number = match.groups()
            series_key = (series_prefix, extension)
            series_uuid = series_uuid_map.get(series_key)
            if series_uuid is None:
                series_uuid = str(uuid.uuid4())
                series_uuid_map[series_key] = series_uuid

            # Keep the original numbering (including zero padding) for order.
            return f"{series_uuid}_{slice_number}{extension}"

        # Non-slice files keep pure UUID naming.
        new_name = str(uuid.uuid4()) + extension

        return new_name
    
    def log_rename(self, old_name: str, new_name: str) -> None:
        """Logs the mapping of an old file name to a new file name in the CSV file.

        Args:            
            old_name (str): The original file name.
            new_name (str): The new file name.

        Returns:
            None
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([old_name, new_name])