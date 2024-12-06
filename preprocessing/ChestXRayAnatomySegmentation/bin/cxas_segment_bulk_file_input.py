#!/usr/bin/env python
import argparse
import logging
import os
from pathlib import Path

from cxas import CXAS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment 159 anatomical structures in X-Ray images.",
        epilog="Written by Constantin Seibold. If you use this tool, please cite accordingly."
    )

    parser.add_argument(
        "-i", "--input",
        metavar="filepath",
        type=str,
        required=True,
        help="Path to a file, directory, or a text file containing paths."
    )

    parser.add_argument(
        "-ot", "--output_type",
        choices=["json", "npy", "npz", "jpg", "png", "dicom-seg"],
        default='jpg',
        help="Storage type of segmentations if they are stored."
    )

    parser.add_argument(
        "-g", "--gpus",
        default="0",
        help="Select specific GPU/CPU to process the input."
    )

    parser.add_argument(
        "-m", "--model",
        choices=["UNet_ResNet50_default"],
        default="UNet_ResNet50_default",
        help="Model used for inference."
    )

    return parser.parse_args()

def process_path(model, path: Path, output_type: str) -> None:
    """Process a single file or directory."""
    if path is not None:
        logging.info(f"Processing file: {path}...")
        model.generate_mask_file(
            filename=str(path),
            create=True,
            do_store=True,
            storage_type=output_type,
        )
    else:
        logging.error(f"{path} is neither a file nor a directory.")

def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    model = CXAS(model_name=args.model, gpus=args.gpus)
    print("Model loaded")
    basepath="/data/mount2/mimic-cxr-jpg-2.1.0.physionet.org"

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix == '.txt':
        logging.info(f"Processing input file list from: {input_path}")
        with open(input_path, 'r') as f:
            for line in f:
                path = Path(line.strip())
                fpath = os.path.join(basepath,path)
                if fpath is not None:
                    process_path(model, fpath, args.output_type)
                else:
                    logging.warning(f"Path does not exist: {path}")
    elif input_path.exists():
        process_path(model, input_path, args.output_type)
    else:
        logging.error(f"{input_path} does not exist.")
        raise FileNotFoundError(f"{input_path} does not exist.")

if __name__ == "__main__":
    main()
