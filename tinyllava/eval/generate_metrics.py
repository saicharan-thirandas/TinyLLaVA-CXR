import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')

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
    i=0
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            data1 = json.loads(line1)
            data2 = json.loads(line2)
            
            # Assert if question_id matches
            print("testing questionid" + str(data1["question_id"]))
            assert data1["question_id"] == data2["question_id"], (
                f"Mismatch in question_id: {data1['question_id']} != {data2['question_id']}"
            )
            
            # Compare texts and compute metrics
            metrics = compare_texts(data1["text"], data2["text"])
            results.append({
                "question_id": data1["question_id"],
                "metrics": metrics
            })
            i=i+1
            if i<1000:
                break
    
    # Write results to output JSON
    with open(output_file, 'w') as out:
        json.dump(results, out, indent=4)
    print(f"Metrics comparison completed. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Replace with your file paths
    file1 = "evaluations/con-and-llm-full-reports.jsonl"
    file2 = "annotations/mimic_reports_test_images_answers.jsonl"
    compare_jsonl_files(file1, file2)
