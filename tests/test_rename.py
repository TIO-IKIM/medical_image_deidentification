import csv
import os
import re
import tempfile
import unittest

from mede.rename import Rename


class TestRename(unittest.TestCase):
    def test_generate_new_name_preserves_slice_number_and_series_uuid(self):
        rename = Rename(input_path=".", output_path=tempfile.mkdtemp())
        series_uuid_map = {}

        first = rename.generate_new_name("scan_001.png", series_uuid_map)
        second = rename.generate_new_name("scan_002.png", series_uuid_map)
        third = rename.generate_new_name("scan_003.png", series_uuid_map)

        pattern = re.compile(
            r"^(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(?P<slice>\d+)\.png$"
        )

        first_match = pattern.match(first)
        second_match = pattern.match(second)
        third_match = pattern.match(third)

        self.assertIsNotNone(first_match)
        self.assertIsNotNone(second_match)
        self.assertIsNotNone(third_match)

        self.assertEqual(first_match.group("uuid"), second_match.group("uuid"))
        self.assertEqual(second_match.group("uuid"), third_match.group("uuid"))

        self.assertEqual(first_match.group("slice"), "001")
        self.assertEqual(second_match.group("slice"), "002")
        self.assertEqual(third_match.group("slice"), "003")

    def test_generate_new_name_non_slice_uses_plain_uuid(self):
        rename = Rename(input_path=".", output_path=tempfile.mkdtemp())

        new_name = rename.generate_new_name("report_final.txt", {})

        self.assertRegex(
            new_name,
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.txt$",
        )

    def test_rename_files_moves_and_logs_mapping(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            input_files = ["seriesA_001.png", "seriesA_002.png", "note.txt"]
            for filename in input_files:
                with open(os.path.join(input_dir, filename), "w", encoding="utf-8") as f:
                    f.write("dummy")

            rename = Rename(input_path=input_dir, output_path=output_dir)
            rename.rename_files()

            # Input files should have been moved out of the input directory.
            self.assertEqual(os.listdir(input_dir), [])

            output_files = os.listdir(output_dir)
            self.assertIn("rename_mapping.csv", output_files)

            moved_files = sorted([name for name in output_files if name != "rename_mapping.csv"])
            self.assertEqual(len(moved_files), 3)

            series_pattern = re.compile(
                r"^(?P<uuid>[0-9a-f-]{36})_(?P<slice>001|002)\.png$"
            )
            series_files = [name for name in moved_files if name.endswith(".png")]
            self.assertEqual(len(series_files), 2)

            first_series_match = series_pattern.match(series_files[0])
            second_series_match = series_pattern.match(series_files[1])
            self.assertIsNotNone(first_series_match)
            self.assertIsNotNone(second_series_match)
            self.assertEqual(
                first_series_match.group("uuid"), second_series_match.group("uuid")
            )

            txt_files = [name for name in moved_files if name.endswith(".txt")]
            self.assertEqual(len(txt_files), 1)
            self.assertRegex(
                txt_files[0],
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.txt$",
            )

            with open(
                os.path.join(output_dir, "rename_mapping.csv"),
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0], ["old_name", "new_name"])
            self.assertEqual(len(rows), 4)

            logged_old_names = sorted(row[0] for row in rows[1:])
            self.assertEqual(logged_old_names, sorted(input_files))


if __name__ == "__main__":
    unittest.main()
