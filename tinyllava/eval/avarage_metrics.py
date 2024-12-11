# Define the script to read input from a file and calculate average metrics
import json
# File name
file_name = "metrics_results.json"

# Read data from the file
with open(file_name, "r") as file:
    data = json.load(file)

# Initialize a dictionary to store the sum of metrics
metric_sums = {}
metric_counts = {}

# Aggregate metrics from the data
for entry in data:
    metrics = entry["metrics"]
    for key, value in metrics.items():
        if key not in metric_sums:
            metric_sums[key] = 0
            metric_counts[key] = 0
        metric_sums[key] += value
        metric_counts[key] += 1

# Calculate average scores for each metric
average_metrics = {metric: metric_sums[metric] / metric_counts[metric] for metric in metric_sums}

# Print the average metrics
print(str(average_metrics))
