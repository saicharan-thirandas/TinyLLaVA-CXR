import json

# Load JSON data from the file
with open('/data/annotations/mimic-cxr-jpg/mimicCxrQAwithReportFormat.json', 'r') as file:
    data = json.load(file)

print(f"'mimicCxrQAwithReportFormat.txt'. Total kept: {len(data)}")

'''
# Filter out the JSON objects with specific criteria
filtered_data_1 = [
    item for item in data
    if  not any(conv["from"] == "gpt" and ((conv["value"].strip().lower()) == "yes" or   (conv["value"].strip().lower()) == "no"  )   for conv in item["conversations"])
]


filtered_data = [
    item for item in filtered_data_1
    if   any(conv["from"] == "gpt" and (len(conv["value"].strip().lower())  <305 ) and  (len(conv["value"].strip().lower()) >300)  for conv in item["conversations"])
]
'''
filtered_data = [
    item for item in data
    if   any(conv["from"] == "human" and "Generate a report" in conv["value"]  for conv in item["conversations"])
]
# Save the filtered data back to a file or print it
with open('/data/annotations/mimic-cxr-jpg/mimic_reports.json', 'w') as outfile:
    json.dump(filtered_data, outfile, indent=4)

print(f"Filtered JSON objects saved to 'filtered_json.txt'. Total kept: {len(filtered_data)}")
