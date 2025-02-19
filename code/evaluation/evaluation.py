import json
import os
import argparse

def calculate_metrics(file_path, dataset_name):
    total_true = 0
    correct_generated_label_true = 0
    total_false = 0
    correct_generated_label_false = 0
    
    total_samples = 0
    total_correct = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                generated_label = str(entry.get("final_label"))
                label = entry.get("label")
                
                total_samples += 1
                if generated_label == str(label):
                    total_correct += 1

                if label == -1:
                    total_true += 1
                    if generated_label == "-1":
                        correct_generated_label_true += 1
                else:
                    total_false += 1
                    if generated_label == str(label):
                        correct_generated_label_false += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {line_number}: {line}")
                print(f"JSONDecodeError: {e}")
            except Exception as e:
                print(f"Unexpected error at line {line_number}: {line}")
                print(f"Error: {e}")

    total_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    verifier_accuracy_true = (correct_generated_label_true / total_true * 100) if total_true > 0 else 0.0
    verifier_accuracy_false = (correct_generated_label_false / total_false * 100) if total_false > 0 else 0.0
    if verifier_accuracy_true + verifier_accuracy_false > 0:
        f1_score = (2 * verifier_accuracy_true * verifier_accuracy_false) / (verifier_accuracy_true + verifier_accuracy_false)
    else:
        f1_score = 0.0

    print(f"\nResults for {dataset_name}:")
    print(f"Total Accuracy: {total_accuracy:.1f}%")
    print(f"Verifier Accuracy for final_answer_correct=True: {verifier_accuracy_true:.1f}%")
    print(f"Verifier Accuracy for final_answer_correct=False: {verifier_accuracy_false:.1f}%")
    print(f"F1 Score: {f1_score:.1f}%")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics from verified results')
    parser.add_argument('--model', type=str, required=True, help='Model name used in the output file names')
    parser.add_argument('--output_dic', type=str, required=True, help='Directory containing the verified results from the second code')
    args = parser.parse_args()
    
    datasets = ["gsm8k", "math", "olym", "omni", "prm", "mathcheck"]
    base_path = args.output_dic
    print("Starting metrics calculation for all datasets...")
    for dataset in datasets:
        file_path = os.path.join(base_path, f"inf_verified_results_{args.model}_{dataset}.jsonl")
        calculate_metrics(file_path, dataset)
    
    print("\nAll calculations completed!")

if __name__ == "__main__":
    main()
