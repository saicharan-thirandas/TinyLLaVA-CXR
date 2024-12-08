import os
import time
import google.generativeai as genai
import random

# Configure the Gemini API with the provided API key
genai.configure(api_key="<API KEY>")
model = genai.GenerativeModel("gemini-1.5-flash-8b")

# Define input and output folders
input_folder = "/data/annotations/mimic-cxr-jpg/mimic-cxr-reports/files/p10"
output_folder = input_folder.replace("mimic-cxr-reports", "mimic-cxr-reports-qa")

# Prompt template
prompt_template = """
You are an AI assistant specialized in biomedical topics. Given a medical report, generate five question-answer pairs focused on key visual aspects of the report.
- The questions should be diverse, covering spatial details, abnormalities, modalities, and implications observed in the report.
- Avoid quoting specific figures or captions directly; instead, refer to the observations as inferred from the image or context provided.
- Ensure answers are concise, clear, and relevant to the corresponding question, avoiding overconfidence.
Generate 5 questions and answers from below text in the exact format:
Q1:<question> && A1:<answer>
Q2:<question> && A2:<answer>
Q3:<question> && A3:<answer>
Q4:<question> && A4:<answer>
Q5:<question> && A5:<answer>
Here is the report: 
{}
"""

# Function to process files
def process_file(input_file_path, output_file_path):
    # Check if the report file already exists
    if os.path.exists(output_file_path):
        print(f"Output file already exists for {input_file_path}. Skipping...")
        return

    # Read the content of the input file
    with open(input_file_path, 'r') as f:
        report_text = f.read()
    
    # Generate the prompt
    prompt = prompt_template.format(report_text)
    
    # Generate output
    print(f"Generating output for {input_file_path}...")
    start_time = time.time()
    try:
        # Use Google's generative model for content generation
        response = model.generate_content( prompt )
        output = response.text
        print(str(output))
        #time.sleep(2000)
    except Exception as e:
        print(f"Error generating output for {input_file_path}: {e}")
        return

    end_time = time.time()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Save the output to the output file
    with open(output_file_path, 'w') as f:
        f.write(output)
    
    print(f"Output saved to {output_file_path}. Execution Time: {end_time - start_time:.2f} seconds")

# Iterate through all files in the input folder
for root, dirs, files in os.walk(input_folder):
    #random.shuffle(files)
    #random.shuffle(dirs)
    for file in files:
        # Skip files that are already output reports
        if file.endswith("-qa.txt"):
            continue
        
        # Construct input file path
        input_file_path = os.path.join(root, file)
        
        # Construct output file path by replacing folder name and appending '-qa.txt' to the file name
        relative_path = os.path.relpath(root, input_folder)
        output_root = os.path.join(output_folder, relative_path)
        output_file_name = os.path.splitext(file)[0] + "-qa.txt"
        output_file_path = os.path.join(output_root, output_file_name)
        
        # Process the file
        process_file(input_file_path, output_file_path)
