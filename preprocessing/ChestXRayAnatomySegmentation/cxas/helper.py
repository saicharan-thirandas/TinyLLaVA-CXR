import torch
from typing import List, Union


def get_available_devices() -> List[str]:
    """
    Get a list of available devices for PyTorch.

    Returns:
        List[str]: A list of strings representing the available devices,
        which may include 'cpu', 'mps', and 'cuda:X' for available CUDA devices.
    """
    devices = []

    # Check for CPU (always available)
    devices.append("cpu")

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")

    # Check for MPS (Apple Silicon GPUs on M1/M2 Macs)
    if torch.backends.mps.is_available():
        devices.append("mps")

    return devices


def find_max_overlap(
    input_string: str, string_list: List[str]
) -> Union[int, List[int]]:
    """
    Find the index of the string with the maximum overlap with the input string.

    Args:
        input_string (str): The string to compare against.
        string_list (List[str]): A list of strings to compare with.

    Returns:
        Union[int, List[int]]: The index of the string with the maximum overlap and
        a list of overlap counts for debugging purposes.
    """

    def overlap_count(s1: str, s2: str) -> int:
        """Helper function to count the number of overlapping characters between two strings."""
        return len(set(s1) & set(s2))

    overlap_counts = [overlap_count(input_string, s) for s in string_list]
    argmax_index = overlap_counts.index(max(overlap_counts)) if overlap_counts else None

    return argmax_index, overlap_counts


def set_gpus(user_input: str) -> Union[str, List[str]]:
    """
    Map user input to the best fitting available GPUs.

    Args:
        user_input (str): A string representing user input which can be a
                          device name, numeric index, or a comma-separated list.

    Returns:
        Union[str, List[str]]: A single device string if one device is mapped,
                                or a list of device strings if multiple devices are mapped.
    """
    available_devices = get_available_devices()

    # Check for direct match first
    if user_input in available_devices:
        return user_input

    # Handle cases like '0', 'cuda:0', '0,1,2', etc.
    if isinstance(user_input, str):
        # Split the user input by commas to handle multi-GPU selections like "0,1,2"
        inputs = user_input.split(",")
        mapped_devices = []

        for inp in inputs:
            inp = inp.strip()  # Strip any whitespace

            # Handle numeric inputs like '0', '1', etc.
            if inp.isdigit():
                index = int(inp)
                if 0 <= index < len(available_devices):
                    # Map single number to corresponding GPU (e.g., 0 -> cuda:0)
                    mapped_devices.append(f"cuda:{index}")
                else:
                    # Fallback for invalid numbers, map to the closest device
                    argmax_index, _ = find_max_overlap(
                        f"cuda:{index}", available_devices
                    )
                    mapped_devices.append(available_devices[argmax_index])
            else:
                # Handle direct text like "cpu", "mps", "cuda:0"
                if inp in available_devices:
                    mapped_devices.append(inp)
                else:
                    # If input is not directly found, find closest matching device
                    argmax_index, _ = find_max_overlap(inp, available_devices)
                    mapped_devices.append(available_devices[argmax_index])

        if len(mapped_devices) == 1:
            return mapped_devices[0]  # Return single device string if only one
        return mapped_devices  # Return list of devices if multiple

    return "cpu"  # Default to CPU if input is invalid or not understood
