import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch


def _create_torch_stub() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")
    torch_stub.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False, allow_tf32=False),
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
    )
    torch_stub.autograd = types.SimpleNamespace(set_detect_anomaly=lambda *_args: None)
    torch_stub.set_num_threads = lambda *_args: None
    return torch_stub


def _load_deidentify_module():
    sys.modules.pop("mede.deidentify", None)

    skullstrip_stub = types.ModuleType("mede.dicom_skullstrip_defacing")
    skullstrip_stub.Inference = object

    dicom_stub = types.ModuleType("mede.dicom_deidentification")
    dicom_stub.DicomDeidentifier = object

    text_stub = types.ModuleType("mede.text_detection")
    text_stub.TextRemoval = object

    wsi_stub = types.ModuleType("mede.wsi_deidentification")
    wsi_stub.WSIDeidentifier = object

    twix_stub = types.ModuleType("mede.twix_deidentification")
    twix_stub.anonymize_twix = lambda *_args: None

    with patch.dict(
        sys.modules,
        {
            "torch": _create_torch_stub(),
            "mede.dicom_skullstrip_defacing": skullstrip_stub,
            "mede.dicom_deidentification": dicom_stub,
            "mede.text_detection": text_stub,
            "mede.wsi_deidentification": wsi_stub,
            "mede.twix_deidentification": twix_stub,
        },
    ):
        return importlib.import_module("mede.deidentify")


class TestDeidentifyCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deidentify = _load_deidentify_module()

    def test_full_pipeline_executes_all_enabled_features(self):
        deidentify = self.deidentify

        dicom_instance = MagicMock()
        wsi_instance = MagicMock()
        skullstrip_instance = MagicMock()
        deface_instance = MagicMock()
        text_instance = MagicMock()

        argv = [
            "deidentify.py",
            "--verbose",
            "--input",
            "input_dir",
            "--output",
            "output_dir",
            "--gpu",
            "2",
            "--skull_strip",
            "--deface",
            "--twix",
            "--wsi",
            "--text-removal",
            "--processes",
            "4",
            "--deidentification-profile",
            "basicProfile",
            "rtnUIDsOpt",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch.object(deidentify, "DicomDeidentifier", return_value=dicom_instance) as dicom_cls,
            patch.object(deidentify, "WSIDeidentifier", return_value=wsi_instance) as wsi_cls,
            patch.object(
                deidentify,
                "Inference",
                side_effect=[skullstrip_instance, deface_instance],
            ) as inference_cls,
            patch.object(deidentify, "TextRemoval", return_value=text_instance) as text_cls,
            patch.object(deidentify, "anonymize_twix") as anonymize_twix,
            patch.object(
                deidentify.torch.autograd,
                "set_detect_anomaly",
            ) as set_detect_anomaly,
            patch.object(deidentify.torch, "set_num_threads") as set_num_threads,
        ):
            deidentify.main()

        self.assertTrue(deidentify.torch.backends.cudnn.benchmark)
        self.assertTrue(deidentify.torch.backends.cudnn.allow_tf32)
        self.assertTrue(deidentify.torch.backends.cuda.matmul.allow_tf32)
        set_detect_anomaly.assert_called_once_with(True)
        set_num_threads.assert_called_once_with(4)

        dicom_cls.assert_called_once_with(
            ["basicProfile", "rtnUIDsOpt"],
            processes=4,
            out_path="output_dir",
            verbose=True,
        )
        dicom_instance.assert_called_once_with("input_dir")

        wsi_cls.assert_called_once_with(verbose=True, out_path="output_dir")
        wsi_instance.assert_called_once_with("output_dir")

        inference_cls.assert_has_calls(
            [
                call(output_path="output_dir", gpu=2, skullstrip=True, verbose=True),
                call(output_path="output_dir", gpu=2, deface=True, verbose=True),
            ]
        )
        skullstrip_instance.assert_called_once_with("output_dir")
        deface_instance.assert_called_once_with("output_dir")

        anonymize_twix.assert_called_once_with("output_dir", "output_dir")

        text_cls.assert_called_once_with(output_path="output_dir", verbose=True)
        text_instance.assert_called_once_with("output_dir")

    def test_no_profile_logs_info_and_skips_metadata_anonymization(self):
        deidentify = self.deidentify

        argv = ["deidentify.py", "--input", "input_file", "--output", "output_dir"]

        with (
            patch.object(sys, "argv", argv),
            patch.object(deidentify, "DicomDeidentifier") as dicom_cls,
            patch.object(deidentify, "WSIDeidentifier") as wsi_cls,
            patch.object(deidentify, "Inference") as inference_cls,
            patch.object(deidentify, "TextRemoval") as text_cls,
            patch.object(deidentify, "anonymize_twix") as anonymize_twix,
            patch.object(deidentify.logging, "info") as log_info,
            patch.object(deidentify.torch, "set_num_threads") as set_num_threads,
        ):
            deidentify.main()

        dicom_cls.assert_not_called()
        wsi_cls.assert_not_called()
        inference_cls.assert_not_called()
        text_cls.assert_not_called()
        anonymize_twix.assert_not_called()
        set_num_threads.assert_called_once_with(1)
        log_info.assert_called_once_with(
            "No DICOM deidentification profile specified. No Metadata anonymization will be performed!"
        )

    def test_boolean_optional_flags_can_disable_a_feature(self):
        deidentify = self.deidentify

        argv = [
            "deidentify.py",
            "--input",
            "input_dir",
            "--output",
            "output_dir",
            "--skull_strip",
            "--no-skull_strip",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch.object(deidentify, "Inference") as inference_cls,
            patch.object(deidentify.logging, "info"),
        ):
            deidentify.main()

        inference_cls.assert_not_called()

    def test_invalid_profile_raises_system_exit(self):
        deidentify = self.deidentify
        argv = [
            "deidentify.py",
            "--deidentification-profile",
            "notAValidProfile",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                deidentify.main()


if __name__ == "__main__":
    unittest.main()