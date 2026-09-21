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
    torch_stub.autograd = types.SimpleNamespace(set_detect_anomaly=MagicMock())
    torch_stub.set_num_threads = MagicMock()
    return torch_stub


def _load_deidentify_module():
    sys.modules.pop("mede.deidentify", None)
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

        torch_stub = _create_torch_stub()
        dicom_stub = types.ModuleType("mede.dicom_deidentification")
        dicom_cls = MagicMock(return_value=dicom_instance)
        dicom_stub.DicomDeidentifier = dicom_cls

        wsi_stub = types.ModuleType("mede.wsi_deidentification")
        wsi_cls = MagicMock(return_value=wsi_instance)
        wsi_stub.WSIDeidentifier = wsi_cls

        skullstrip_stub = types.ModuleType("mede.dicom_skullstrip_defacing")
        inference_cls = MagicMock(
            side_effect=[skullstrip_instance, deface_instance]
        )
        skullstrip_stub.Inference = inference_cls

        text_stub = types.ModuleType("mede.text_detection")
        text_cls = MagicMock(return_value=text_instance)
        text_stub.TextRemoval = text_cls

        twix_stub = types.ModuleType("mede.twix_deidentification")
        anonymize_twix = MagicMock()
        twix_stub.anonymize_twix = anonymize_twix

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

        with patch.dict(
            sys.modules,
            {
                "torch": torch_stub,
                "mede.dicom_deidentification": dicom_stub,
                "mede.dicom_skullstrip_defacing": skullstrip_stub,
                "mede.text_detection": text_stub,
                "mede.wsi_deidentification": wsi_stub,
                "mede.twix_deidentification": twix_stub,
            },
        ), patch.object(sys, "argv", argv):
            deidentify.main()

        self.assertTrue(torch_stub.backends.cudnn.benchmark)
        self.assertTrue(torch_stub.backends.cudnn.allow_tf32)
        self.assertTrue(torch_stub.backends.cuda.matmul.allow_tf32)
        torch_stub.autograd.set_detect_anomaly.assert_called_once_with(True)
        torch_stub.set_num_threads.assert_called_once_with(4)

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

        text_cls.assert_called_once_with(
            output_path="output_dir", verbose=True, interactive=False
        )
        text_instance.assert_called_once_with("output_dir")

    def test_no_profile_logs_info_and_skips_metadata_anonymization(self):
        deidentify = self.deidentify

        argv = ["deidentify.py", "--input", "input_file", "--output", "output_dir"]

        with patch.object(sys, "argv", argv), patch.object(
            deidentify.logging, "info"
        ) as log_info:
            deidentify.main()

        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("mede.dicom_deidentification", sys.modules)
        self.assertNotIn("mede.wsi_deidentification", sys.modules)
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

        with patch.object(sys, "argv", argv), patch.object(
            deidentify.logging, "info"
        ):
            deidentify.main()

        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("mede.dicom_skullstrip_defacing", sys.modules)

    def test_help_does_not_import_optional_dependencies(self):
        deidentify = self.deidentify
        optional_modules = [
            "torch",
            "numpy",
            "pydicom",
            "cv2",
            "easyocr",
            "mede.dicom_deidentification",
            "mede.dicom_skullstrip_defacing",
            "mede.text_detection",
            "mede.wsi_deidentification",
            "mede.twix_deidentification",
        ]
        sentinel = object()
        previous = {name: sys.modules.get(name, sentinel) for name in optional_modules}
        for name in optional_modules:
            sys.modules.pop(name, None)

        try:
            with patch.object(sys, "argv", ["deidentify.py", "--help"]):
                with self.assertRaises(SystemExit) as exit_context:
                    deidentify.main()
            self.assertEqual(exit_context.exception.code, 0)
            for name in optional_modules:
                self.assertNotIn(name, sys.modules)
        finally:
            for name, module in previous.items():
                if module is sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

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
