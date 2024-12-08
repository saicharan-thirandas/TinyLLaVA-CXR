import os
import json
import random

def process_text_file(txt_file_path):
    """
    Parse the text file and extract questions and answers.
    """
    if not os.path.exists(txt_file_path):
        print(str(txt_file_path) + "file not present")
        return None


    try:
        with open(txt_file_path, "r") as file:
            content = file.read()
        qa_pairs = content.strip().split("\n")
        conversations = []
        for i, qa in enumerate(qa_pairs):
            if "&&" in qa:
                question, answer = qa.split("&&")
                question = question.split(":")[1].strip()
                answer = answer.split(":")[1].strip()

                # Randomly decide placement of <image> for the first question
                if i == 0:
                    question = random.choice([f"<image>\n{question}", f"{question}\n<image>"])

                conversations.append({"from": "human", "value": question})
                conversations.append({"from": "gpt", "value": answer})
        return conversations
    except Exception as e:
        print(f"Error parsing {txt_file_path}: {e}")
        return None

def process_json_file(input_json_path, base_path, output_json_path):
    """
    Read the input JSON, process text files, and update conversations.
    """
    try:
        with open(input_json_path, "r") as json_file:
            data = json.load(json_file)

        updated_data = []
        for entry in data:
            image_path = entry.get("image", "")
            # Extract up to the second-to-last directory and replace .jpg with .txt
            txt_file_path = os.path.join(base_path, "/".join(image_path.split("/")[:-1]) + "-qa.txt")

            new_conversations = process_text_file(txt_file_path)

            if new_conversations:
                entry["conversations"] = new_conversations
            
            updated_data.append(entry)

        with open(output_json_path, "w") as json_file:
            json.dump(updated_data, json_file, indent=4)

        print(f"Updated JSON saved to {output_json_path}")

    except Exception as e:
        print(f"Error processing JSON file: {e}")

if __name__ == "__main__":
    input_json_path = "mimic_reports_test_images.json"
    output_json_path = "mimic_conversation_test_images.json"
    base_path = "mimic-cxr-reports-qa"  # Set this to the base directory containing the image and text files

    process_json_file(input_json_path, base_path, output_json_path)
