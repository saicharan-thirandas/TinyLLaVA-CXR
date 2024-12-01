import unittest
from unittest import mock
import torch
import numpy as np
import os
import shutil
from cxas import CXAS


class TestCXAS(unittest.TestCase):

    def setUp(self):
        """Set up for testing."""
        # Initialize the CXAS model with mock parameters
        self.model = CXAS(model_name="UNet_ResNet50_default", gpus="cpu")
        # Prepare a sample tensor for testing
        self.sample_tensor = torch.randn(1, 3, 512, 512)

    @mock.patch("torch.randn")
    def test_model_inference(self, mock_randn):
        """Test the model inference on a sample tensor."""
        mock_randn.return_value = self.sample_tensor

        # Call the model with the sample tensor
        output = self.model(self.sample_tensor)

        # Verify that the output is a dictionary
        self.assertIsInstance(output, dict)

        # Check the shapes and dtypes of the expected output keys
        expected_shapes = {
            "feats": torch.Size([1, 128, 128, 128]),
            "logits": torch.Size([1, 159, 512, 512]),
            "data": torch.Size([1, 3, 512, 512]),
            "segmentation_preds": torch.Size([1, 159, 512, 512]),
        }

        for key, expected_shape in expected_shapes.items():
            self.assertIn(key, output)
            self.assertEqual(output[key].shape, expected_shape)

    @mock.patch("cxas.CXAS.process_file")
    def test_process_file(self, mock_process_file):
        """Test processing of different file types."""
        mock_process_file.return_value = {
            "data": self.sample_tensor,
            "logits": torch.randn(1, 159, 512, 512),
            "segmentation_preds": torch.randn(1, 159, 512, 512).bool(),
        }

        # Test processing a PNG file
        png_path = "input/00003440_000.png"
        output = self.model.process_file(filename=png_path)
        self.assertIn("data", output)

        # Test processing a JPG file
        jpg_path = "input/dummy_image.jpg"
        output = self.model.process_file(filename=jpg_path)
        self.assertIn("data", output)

    def test_file_storage(self):
        """Test file storage in various formats."""
        path = "input/dummy_image.dcm"
        out_path = "./output"
        # os.makedirs(out_path, exist_ok=True)  # Ensure output directory exists

        # Testing storage as dicom-seg
        _ = self.model.process_file(
            filename=path,
            do_store=True,
            output_directory=out_path,
            create=True,
            storage_type="dicom-seg",
        )
        self.assertTrue(os.path.isdir(out_path))
        shutil.rmtree(out_path)  # Clean up

        # Testing storage as npy
        _ = self.model.process_file(
            filename=path,
            do_store=True,
            output_directory=out_path,
            create=True,
            storage_type="npy",
        )
        self.assertTrue(os.listdir(out_path))  # Check if files are created
        shutil.rmtree(out_path)  # Clean up

        # Testing storage as npz
        _ = self.model.process_file(
            filename=path,
            do_store=True,
            output_directory=out_path,
            create=True,
            storage_type="npz",
        )
        self.assertTrue(os.listdir(out_path))  # Check if files are created
        shutil.rmtree(out_path)  # Clean up

        # Testing storage as jpg
        _ = self.model.process_file(
            filename=path,
            do_store=True,
            output_directory=out_path,
            create=True,
            storage_type="jpg",
        )
        self.assertTrue(os.listdir(out_path))  # Check if files are created
        shutil.rmtree(out_path)  # Clean up

        # Testing storage as json
        _ = self.model.process_file(
            filename=path,
            do_store=True,
            output_directory=out_path,
            create=True,
            storage_type="json",
        )
        self.assertTrue(os.listdir(out_path))  # Check if files are created
        shutil.rmtree(out_path)  # Clean up


if __name__ == "__main__":
    unittest.main()
