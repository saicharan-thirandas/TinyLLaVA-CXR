from PIL import Image
import SimpleITK as sitk
import numpy as np
import os
import json
import torch
import torch.nn.functional as F
from itertools import groupby
from cxas.label_mapper import id2label_dict, category_ids
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from cxas.io_utils.dicomseg_2d import write_dicom_seg
from cxas.io_utils.create_annotations import (
    get_coco_json_format,
    create_category_annotation,
)
from cxas.io_utils.mask_to_coco import toBox, mask_to_annotation
from pathlib import Path

this_directory = Path(__file__).parent


class FolderDataset(Dataset):
    """
    Dataset class to load images from a folder.
    """

    def __init__(self, path: str, gpus: str):
        """
        Initialize the FolderDataset.

        Args:
            path (str): Path to the folder containing images.
            gpus (str): GPU(s) to use for processing.
        """
        super(Dataset, self).__init__()
        file_types = ["jpg", "png", "dcm"]
        self.fileloader = FileLoader("")
        self.files = [
            os.path.join(path, i)
            for i in os.listdir(path)
            if i.split(".")[-1].lower() in file_types
        ]

    def collate_fn(self, batch):
        """
        Custom collate function for batching.

        Args:
            batch: List of samples to batch.

        Returns:
            dict: Batched data.
        """
        out_dict = {}
        for b in batch:
            for k in b.keys():
                if k in out_dict.keys():
                    out_dict[k] += [b[k]]
                else:
                    out_dict[k] = [b[k]]

        for k in out_dict.keys():
            if k == "data":
                out_dict[k] = torch.cat(out_dict[k], dim=0)

        return out_dict

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            dict: Sample data.
        """
        return self.fileloader.load_file(self.files[index])

    def __len__(
        self,
    ):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.files)


def get_folder_loader(
    path: str, gpus: str, batch_size: int
) -> torch.utils.data.DataLoader:
    """
    Get DataLoader for a folder dataset.

    Args:
        path (str): Path to the folder containing images.
        gpus (str): GPU(s) to use for processing.
        batch_size (int): Batch size.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the folder dataset.
    """
    dataset = FolderDataset(path, gpus)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )
    return loader


class FileLoader:
    """
    Class to load files (images or DICOM) from disk.
    """

    def __init__(self, gpus: str):
        """
        Initialize the FileLoader.

        Args:
            gpus (str): GPU(s) to use for processing.
        """
        self.base_size = 512
        self.file_types = {
            "jpg": self.load_image,
            "png": self.load_image,
            "dcm": self.load_dicom,
        }

        if "cpu" in gpus:
            self.gpus = "cpu"
        else:
            self.gpus = [int(i) for i in gpus.split(",") if len(i) > 0]

    def to_gpu(self, array: torch.tensor) -> torch.tensor:
        """
        Move tensor to GPU.

        Args:
            array (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Tensor moved to GPU if available.
        """
        if len(self.gpus) > 0 and self.gpus != "cpu":
            assert torch.cuda.is_available()
            return array.to(torch.device("cuda:{}".format(self.gpus[0])))
        else:
            return array

    def normalize(self, array: torch.tensor) -> torch.tensor:
        """
        Normalize image tensor.

        Args:
            array (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Normalized tensor.
        """
        assert array.shape[0] == 3, f"{array.shape}"
        # ImageNet normalization
        # Array to be assumed in range [0,1]
        assert (array.min() >= 0) and (array.max() <= 1)

        array = (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
            torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        )
        return array

    def load_file(self, file_path: str) -> dict:
        """
        Load file based on its extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            dict: Loaded file data.
        """
        assert file_path.split(".")[-1].lower() in list(
            self.file_types.keys()
        ), f"filetype not supported: {file_path.split('.')[-1].lower()}"

        return self.file_types[file_path.split(".")[-1].lower()](file_path)

    def load_image(self, image_path: str) -> dict:
        """
        Load image from file.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Image data.
        """
        array = np.array(Image.open(image_path).convert(mode="RGB"))
        array = np.transpose(array, [2, 0, 1])
        original_array = np.copy(array)
        orig_file_size = array.shape[-2:]
        array = torch.tensor(array).float() / 255
        array = self.normalize(array)
        array = self.to_gpu(array)
        return {
            "data": F.interpolate(array.unsqueeze(0), self.base_size),
            "orig_data": original_array,
            "filename": image_path,
            "file_size": orig_file_size,
        }

    def load_dicom(self, image_path: str) -> dict:
        """
        Load DICOM image.

        Args:
            image_path (str): Path to the DICOM file.

        Returns:
            dict: DICOM image data.
        """
        image = sitk.ReadImage(image_path)
        array_view = sitk.GetArrayFromImage(image).astype(np.float32)
        assert (len(array_view.shape) == 3) and (array_view.shape[0] == 1)
        orig_file_size = array_view.shape[-2:]
        tensor = torch.tensor(array_view).float()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensor = torch.cat([tensor, tensor, tensor], 0)
        original_array = (tensor.numpy() * 255).astype(np.uint8)
        array = self.normalize(tensor)
        array = self.to_gpu(array)
        return {
            "data": F.interpolate(array.unsqueeze(0), self.base_size),
            "orig_data": original_array,
            "filename": image_path,
            "file_size": orig_file_size,
        }


class FileSaver:
    """
    Class to save prediction results.
    """

    def __init__(
        self,
    ):
        """
        Initialize the FileSaver.
        """
        self.save_modes = {
            "npz": self.export_prediction_as_npz,
            "npy": self.export_prediction_as_numpy,
            "jpg": self.export_prediction_as_jpg,
            "png": self.export_prediction_as_png,
            "json": self.export_prediction_as_json,
            "dicom-seg": self.export_prediction_as_dicomseg,
        }

    def save_prediction(
        self, filename : str, mask: np.array, output_filepath: str, mode: str
    ) -> None:
        """
        Save prediction results.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
            mode (str): Save mode.
        """
        assert mode in list(self.save_modes.keys())
        assert len(mask.shape) == 3
        self.save_modes[mode](filename, mask, output_filepath)

    def export_prediction_as_dicomseg(
        self, mask: np.array, outdir: str, file_name: str
    ) -> None:
        """
        Export prediction as DICOM segmentation.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
        """
        assert (
            os.path.splitext(file_name)[-1] == ".dcm"
        ), "Input file has to be dicom to be stored \
            as dicom-seg, it was --{}--".format(
            os.path.splitext(file_name)[-1]
        )
        out_dir = os.path.join(outdir, os.path.splitext(file_name)[0].split("/")[-1])
        os.makedirs(out_dir, exist_ok=True)

        write_dicom_seg(
            metainfo=os.path.join(str(this_directory), "data/metainfo.json"),
            dcm_file=file_name,
            mask=mask.astype(np.uint8),
            out_dir=out_dir,
            id_label_dict=id2label_dict,
        )

    def export_prediction_as_jpg(
        self, filename: str, mask: np.array, output_file_path: str
    ) -> None:
        """
        Export prediction as JPG images.

        Args:
            filename (str): Path to the input image.
            mask (np.array): Prediction mask.
            output_file_path (str): Output file path.
        """
        from PIL import Image
        import numpy as np

        # Load the image and ensure it's in RGB format
        image = np.array(Image.open(filename))

        # Debugging shapes
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

        # List of indices for the masks of interest
        '''
        mask_of_interest_indices = [
            134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153
        ]
        '''
        mask_of_interest_indices = [134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 
        156, 157, 158, 
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 
        115, 116, 117, 118, 119, 120, 154, 155 ,
        105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 133,
        28, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 
        95, 96, 97, 98, 99, 100, 101, 102, 103, 104 ]



        # Initialize a union mask (assuming binary mask with 0s and 1s)
        union_mask = np.zeros_like(mask[0], dtype=bool)

        # Create a union of the masks of interest
        for i in mask_of_interest_indices:
            union_mask = np.logical_or(union_mask, mask[i])

        # Debugging union mask shape
        print(f"Union mask shape: {union_mask.shape}")

        # Ensure the mask matches the dimensions of the image
        if len(image.shape) == 3:  # Multi-channel image
            union_mask = np.expand_dims(union_mask, axis=-1)  # Add channel dimension

        # Apply the mask to the image
        filtered_image = np.where(union_mask, image, 0).astype(np.uint8)

        # Save the filtered image
        Image.fromarray(filtered_image).save(output_file_path)


    def export_prediction_as_png(
        self, mask: np.array, outdir: str, file_name: str
    ) -> None:
        """
        Export prediction as PNG images.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
        """
        assert len(mask.shape) == 3
        fileroot = os.path.splitext(file_name)[0].split("/")[-1]
        outdir = os.path.join(outdir, fileroot)
        os.makedirs(outdir, exist_ok=True)

        for i in range(len(mask)):
            out_path = os.path.join(outdir, id2label_dict[str(i)] + ".png")
            Image.fromarray(mask[i]).convert("1").save(out_path)

    def export_prediction_as_numpy(
        self, mask: np.array, outdir: str, file_name: str
    ) -> None:
        """
        Export prediction as NumPy arrays.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
        """
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(
            outdir, os.path.splitext(file_name)[0].split("/")[-1] + ".npy"
        )
        np.save(out_path, mask)

    def export_prediction_as_npz(
        self, mask: np.array, outdir: str, file_name: str
    ) -> None:
        """
        Export prediction as compressed NumPy arrays.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
        """
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(
            outdir, os.path.splitext(file_name)[0].split("/")[-1] + ".npz"
        )
        np.savez_compressed(out_path, mask)

    def export_prediction_as_json(
        self,
        mask: np.array,
        outdir: str,
        file_name: str,
        img_id: int = 1,
        base_ann_id: int = 1,
    ) -> None:
        """
        Export prediction as JSON in COCO format.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
            img_id (int, optional): Image ID. Defaults to 1.
            base_ann_id (int, optional): Base annotation ID. Defaults to 1.
        """
        coco_format = get_coco_json_format()
        coco_format["categories"] = create_category_annotation(category_ids)

        coco_format["images"] = []
        coco_format["images"].append([{"id": img_id, "file_name": file_name}])

        coco_format["annotations"] = mask_to_annotation(
            mask=mask, base_ann_id=base_ann_id, img_id=img_id
        )

        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(
            outdir, os.path.splitext(file_name)[0].split("/")[-1] + ".json"
        )

        with open(out_path, "w") as outfile:
            json.dump(coco_format, outfile)
