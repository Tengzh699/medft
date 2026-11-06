import unittest
from types import SimpleNamespace
from unittest.mock import patch

from medft.environment import check_data_backend
from medft.training import train


class DataBackendTests(unittest.TestCase):
    def test_arrow_import_error_is_reported(self):
        def import_backend(name):
            if name == "pyarrow":
                raise ImportError("pyarrow requires NumPy 2.0 or newer, found 1.26.4")
            return SimpleNamespace()

        with patch("medft.environment.import_module", side_effect=import_backend):
            report = check_data_backend()
        self.assertFalse(report["ok"])
        self.assertIn("ImportError", report["error"])
        self.assertIn("NumPy 2.0", report["error"])

    def test_dataset_conversion_is_exercised(self):
        class TinyDataset:
            @staticmethod
            def from_dict(columns):
                return [{"value": value} for value in columns["value"]]

        modules = {
            "numpy": SimpleNamespace(__version__="1.26.4", int64=int, array=lambda values, dtype: values),
            "pyarrow": SimpleNamespace(__version__="19.0.1"),
            "datasets": SimpleNamespace(__version__="3.5.0", Dataset=TinyDataset),
        }
        with patch("medft.environment.import_module", side_effect=modules.__getitem__):
            report = check_data_backend()
        self.assertTrue(report["ok"])
        self.assertEqual(report["rows"], 2)

    def test_conversion_failure_is_reported(self):
        class BrokenDataset:
            @staticmethod
            def from_dict(columns):
                raise RuntimeError("binary conversion failed")

        modules = {
            "numpy": SimpleNamespace(int64=int, array=lambda values, dtype: values),
            "pyarrow": SimpleNamespace(),
            "datasets": SimpleNamespace(Dataset=BrokenDataset),
        }
        with patch("medft.environment.import_module", side_effect=modules.__getitem__):
            self.assertIn("binary conversion failed", check_data_backend()["error"])

    def test_training_stops_before_loading_model_when_backend_is_broken(self):
        with patch("medft.training.check_data_backend", return_value={"ok": False, "error": "incompatible pyarrow"}), \
                patch("medft.training.load_quantized_model") as load_model:
            with self.assertRaisesRegex(RuntimeError, "incompatible pyarrow"):
                train({})
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
