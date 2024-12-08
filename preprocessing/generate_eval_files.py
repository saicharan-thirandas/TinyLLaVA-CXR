import json
import random

# Function to generate question JSONL and answer JSONL
def generate_jsonl(data):
    questions = []
    answers = []

    for entry in data:
        image = entry["image"]
        id_serial = entry["id"]
        for i, conversation in enumerate(entry["conversations"], start=1):
            if conversation["from"] == "human":
                question_text = conversation["value"]
                # Ensure "<image>" is at the beginning if not already present
                if "<image>" not in question_text:
                    question_text =  random.choice([f"<image>\n{question_text}", f"{question_text}\n<image>"])

                # Create question JSON
                questions.append({
                    "question_id": f"{id_serial}_{i}",
                    "image": image,
                    "text": question_text
                })

            if conversation["from"] == "gpt":
                # Create answer JSON
                answers.append({
                    "question_id": f"{id_serial}_{i-1}",
                    "image": image,
                    "prompt": question_text,
                    "text": conversation["value"]
                })

    return questions, answers


# Reading data from an input file
input_filename = "jsons/mimic_conversation_test_images.json"

# Load JSON data from the input file
with open(input_filename, "r") as infile:
    data = json.load(infile)

# Generate JSONL for questions and answers
questions_jsonl, answers_jsonl = generate_jsonl(data)

# Save to JSONL files
questions_filename = input_filename.replace(".json", "_questions.jsonl")
answers_filename = input_filename.replace(".json", "_answers.jsonl")

with open(questions_filename, "w") as qf:
    for question in questions_jsonl:
        qf.write(json.dumps(question) + "\n")  # Write each JSON object as a line

with open(answers_filename, "w") as af:
    for answer in answers_jsonl:
        af.write(json.dumps(answer) + "\n")  # Write each JSON object as a line

questions_filename, answers_filename
