import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('punkt')

# Function to compute BLEU score
def compute_bleu(reference, hypothesis):
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    scores = {
        f"BLEU-{i}": sentence_bleu(reference_tokens, hypothesis_tokens, weights=[1/i]*i)
        for i in range(1, 5)
    }
    return scores

# Function to compute METEOR score
def compute_meteor(reference, hypothesis):
    """Compute METEOR score using nltk."""
    reference_tokens = word_tokenize(reference)  # Tokenize reference
    hypothesis_tokens = word_tokenize(hypothesis)  # Tokenize hypothesis
    return {"METEOR": meteor_score([reference_tokens], hypothesis_tokens)}

# Function to compute ROUGE-L score
def compute_rouge(reference, hypothesis):
    rouge = Rouge()
    if reference == "":
        reference="report"
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return {"ROUGE-L": scores["rouge-l"]["f"]}

# Function to compute overall metrics for two texts
def compare_texts(reference, hypothesis):
    metrics = {}
    metrics.update(compute_bleu(reference, hypothesis))
    metrics.update(compute_meteor(reference, hypothesis))
    metrics.update(compute_rouge(reference, hypothesis))
    return metrics

# Main function to process JSONL files and compare metrics
def compare_jsonl_files(file1, file2, output_file="metrics_results.json"):
    results = []

    # Read and load both files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = [json.loads(line) for line in f1]
        data2 = [json.loads(line) for line in f2]

    # Sort both files by question_id to ensure the order is consistent
    data1_sorted = sorted(data1, key=lambda x: x['question_id'])
    data2_sorted = sorted(data2, key=lambda x: x['question_id'])

    # Compare the texts based on sorted question IDs
    for entry1, entry2 in zip(data1_sorted, data2_sorted):
        assert entry1["question_id"] == entry2["question_id"], (
            f"Mismatch in question_id: {entry1['question_id']} != {entry2['question_id']}"
        )

        # Compare texts and compute metrics
        metrics = compare_texts(entry1["text"], entry2["text"])
        results.append({
            "question_id": entry1["question_id"],
            "metrics": metrics
        })

    # Aggregate metrics from the data
    metric_sums = {}
    metric_counts = {}
    for entry in results:
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
    
    # Write results to output JSON
    with open(output_file, 'w') as out:
        json.dump(average_metrics, out, indent=4)
    print(f"Metrics comparison completed. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Replace with your file paths
    #file1 = "answer/conversation_first_qa_run.jsonl"
    file1 = "answer/llava-med-conversation.jsonl"

    output_file = file1.split("/")[1].split(".")[0] + "_average_metrics.json"
    
    file2 = "annotations/mimic_conversation_test_images_answers.jsonl"
    compare_jsonl_files(file1, file2, output_file)