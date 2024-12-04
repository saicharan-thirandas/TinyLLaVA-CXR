import os
import time
from gpt4all import GPT4All

# Path to the parent folder
parent_folder = "/data/annotations/mimic-cxr-jpg/mimic-cxr-reports-qa/files/"

# Model name
model_name = "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf"

# Prompt template
prompt_template = """
You are a specialized AI assistant in interpreting chest X-ray imagery and biomedical topics.
You will receive the final report of a chest X-ray, including sections like findings, impressions, history, and indications, though sections may vary. The actual X-ray image is not accessible.
A radiologist will be asking you questions about the visual details of the X-ray to help interpret the image. So, be prepared to answer questions about the visual details of the X-ray, such as the presence of specific anatomical structures, abnormalities, or artifacts, etc.
Generate 5 questions and answers from this report in the format Q1:<question> && A1:<answer> in 5 lines, focus on FINDINGS and IMPRESSION: 
Here is the report: 
{}
"""

# Initialize the model
print(f"Initializing model: {model_name}")
model = GPT4All(model_name, device='gpu')

# Function to process files
def process_file(file_path):
    # Determine the base name of the file (without extension)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Define report file path based on the base name
    report_file_path = os.path.join(os.path.dirname(file_path), f"{base_name}_report.txt")
    
    # Check if the report file already exists
    if os.path.exists(report_file_path):
        print(f"Report file already exists for {file_path}. Skipping...")
        return

    # Read the content of the generic file
    with open(file_path, 'r') as f:
        report_text = f.read()
    
    # Generate the prompt
    prompt = prompt_template.format(report_text)
    
    # Generate output
    print(f"Generating output for {file_path}...")
    start_time = time.time()
    with model.chat_session():
        output = model.generate(prompt, max_tokens=300)
    end_time = time.time()
    
    # Save the output to the report file
    with open(report_file_path, 'w') as f:
        f.write(output)
    
    print(f"Output saved to {report_file_path}. Execution Time: {end_time - start_time:.2f} seconds")

# Iterate through all files in the parent folder
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        # Skip files that are already reports
        if file.endswith("_report.txt"):
            continue
        
        # Process other text files
        file_path = os.path.join(root, file)
        process_file(file_path)
