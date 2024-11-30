import json
import os
import pandas as pd

# Define file paths
base_dir = "/data/annotations/mimic-cxr-jpg/"
csv_file_path = os.path.join(base_dir, "mimic-cxr-2.0.0-metadata.csv")
image_filenames_file = os.path.join(base_dir, "image_filenames.txt")
basepath = os.path.join(base_dir, "mimic-cxr-reports")
output_file = os.path.join(base_dir, "mimic_reports_all_images.json")

# Starting ID for JSON entries
id_start = 1000001

# Read the CSV file and create a dictionary grouped by ViewPosition
try:
    df = pd.read_csv(csv_file_path)
    view_dict = df.groupby("ViewPosition")["dicom_id"].apply(list).to_dict()
except FileNotFoundError:
    raise FileNotFoundError(f"Metadata file not found: {csv_file_path}")
except Exception as e:
    raise Exception(f"Error reading metadata CSV: {e}")

# Read image paths from image_filenames.txt
try:
    with open(image_filenames_file, "r") as file:
        image_filenames = [line.strip() for line in file.readlines() if line.strip()]
except FileNotFoundError:
    raise FileNotFoundError(f"Image filenames file not found: {image_filenames_file}")
except Exception as e:
    raise Exception(f"Error reading image filenames: {e}")

# Initialize reports list
reports = []

idx=id_start
# Process each image path
for imagepath in image_filenames:
    try:
        # Extract the dicom_id from the image path
        dicom_id = imagepath.split('/')[-1].split('.')[0]

        # Skip if dicom_id is not in the desired ViewPosition lists
        if dicom_id not in (view_dict.get("PA", []) + view_dict.get("AP", [])):
            continue

        # Construct the corresponding .txt file path
        txt_path = os.path.join(basepath, os.path.dirname(imagepath) + ".txt")

        # Read the content of the .txt file if it exists
        if os.path.exists(txt_path):
            with open(txt_path, "r") as txt_file:
                gpt_value = txt_file.read().strip()
                idx=idx+1
                print(str(idx))
        else:
            continue

        # Create the JSON entry for the image
        report = {
            "id": str(idx),
            "image": imagepath,
            "conversations": [
                {
                    "from": "human",
                    "value": "Generate a report\n<image>"
                },
                {
                    "from": "gpt",
                    "value": gpt_value
                }
            ]
        }
        reports.append(report)

    except Exception as e:
        print(f"Error processing image {imagepath}: {e}")

# Write the JSON list to the output file
try:
    with open(output_file, "w") as json_file:
        json.dump(reports, json_file, indent=4)
    print(f"JSON file generated successfully: {output_file}")
except PermissionError:
    raise PermissionError(f"Permission denied. Cannot write to {output_file}")
except Exception as e:
    raise Exception(f"Error writing JSON output: {e}")
