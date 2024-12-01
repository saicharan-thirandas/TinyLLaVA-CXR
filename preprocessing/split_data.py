import json
import random

# Load the JSON data
input_file = "annotations/mimic-cxr-jpg/mimic_reports_all_images.json"
with open(input_file, "r") as f:
    data = json.load(f)

# Shuffle the data and split
total_samples = len(data)
test_size = int(total_samples * 0.1)

# Randomly select indices for the test set
random.seed(42)  # For reproducibility
indices = list(range(total_samples))
random.shuffle(indices)

test_indices = indices[:test_size]
train_indices = indices[test_size:]

# Create test and train splits
test_data = [data[i] for i in test_indices]
train_data = [data[i] for i in train_indices]

# Save the splits to separate files
with open("annotations/mimic-cxr-jpg/mimic_reports_test_images.json", "w") as test_file:
    json.dump(test_data, test_file, indent=4)

with open("annotations/mimic-cxr-jpg/mimic_reports_train_images.json", "w") as train_file:
    json.dump(train_data, train_file, indent=4)

print(f"Split complete: {len(train_data)} samples in train.json, {len(test_data)} samples in test.json.")
