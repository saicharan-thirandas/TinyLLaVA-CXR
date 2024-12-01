import unittest
from unittest import mock
import logging
import sys
import os
from pathlib import Path

# Adjust the import path based on your project structure
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # Add the parent directory to sys.path


class TestCLI(unittest.TestCase):

    @mock.patch("argparse.ArgumentParser.parse_args")
    def test_parse_arguments(self, mock_parse_args):
        """Test argument parsing."""
        mock_parse_args.return_value = mock.Mock(
            input="/input/dummy_image.jpg",
            output="./output",
            output_type="png",
            gpus="0",
            model="UNet_ResNet50_default",
        )
        args = parse_arguments()
        self.assertEqual(args.input, "/input/dummy_image.jpg")
        self.assertEqual(args.output, "./output")
        self.assertEqual(args.output_type, "png")
        self.assertEqual(args.gpus, "0")
        self.assertEqual(args.model, "UNet_ResNet50_default")

    @mock.patch(
        "bin.cxas_segment.CXAS"
    )  # Adjust to the actual import based on structure
    def test_process_file(self, mock_cxas):
        """Test processing a file."""
        mock_instance = mock_cxas.return_value
        mock_instance.process_file = mock.Mock()

        # Mocking sys.argv to simulate command line arguments
        sys.argv = ["cxas_segment", "-i", "file.jpg", "-o", "./output"]

        main()  # Call the main function

        # Verify that process_file is called with the right arguments
        mock_instance.process_file.assert_called_once_with(
            filename="file.jpg",
            output_directory="./output",
            create=True,
            do_store=True,
            storage_type="png",  # Default output type
        )

    @mock.patch("bin.cxas_segment.CXAS")
    def test_process_directory(self, mock_cxas):
        """Test processing a directory."""
        mock_instance = mock_cxas.return_value
        mock_instance.process_folder = mock.Mock()

        # Mocking sys.argv to simulate command line arguments
        sys.argv = ["cxas_segment", "-i", "./input", "-o", "./output"]

        main()  # Call the main function

        # Verify that process_folder is called with the right arguments
        mock_instance.process_folder.assert_called_once_with(
            input_directory_name="./input",
            output_directory="./output",
            create=True,
            storage_type="png",  # Default output type
        )

    @mock.patch("bin.cxas_segment.logging")
    def test_invalid_input_path(self, mock_logging):
        """Test error handling for invalid input path."""
        with mock.patch("pathlib.Path.is_file", return_value=False), mock.patch(
            "pathlib.Path.is_dir", return_value=False
        ), self.assertRaises(
            SystemExit
        ):  # Catching SystemExit raised by the parser

            # Mocking sys.argv to simulate command line arguments
            sys.argv = ["cxas_segment", "-i", "invalid_path", "-o", "./output"]
            main()

        # Check if the logging error method was called
        mock_logging.error.assert_called_with(
            "invalid_path is neither a file nor a directory."
        )


if __name__ == "__main__":
    unittest.main()
