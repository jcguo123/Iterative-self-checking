import json
import os
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
from typing import List, Dict
from transformers import AutoTokenizer
from collections import Counter
import argparse

torch.cuda.memory._set_allocator_settings('expandable_segments:False')

def process_steps(steps):
    return "\n".join(f"<{i}>{step}</{i}>" for i, step in enumerate(steps))

def extract_answer(solution_text: str):
    if not solution_text:
        return None
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    return matches[-1].strip() if matches else None

class VLLMInference:
    def __init__(self, model_path: str):
        self.model_name = os.path.basename(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_prefix_caching=True,
            swap_space=16,
            max_num_seqs=400
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_tokens=8192,
            seed=42
        )
        
        self.verifier_system_prompt = """You are a secondary verifier for math problem solutions. The first verifier's task was to review math solutions paragraph by paragraph and identify the earliest error (if any), returning -1 if no errors were found.

The first verifier may have made mistakes. Your job is to carefully check their work. You will receive:
1. The original math problem
2. The solution steps
3. The first verifier's generated label (paragraph index where they found the first error, or -1)
4. The first verifier's reasoning

IMPORTANT:
1. You must check each paragraph carefully, as if you are re-solving the problem from scratch.
2. If you find any error—no matter how minor—locate the earliest paragraph containing that error.
3. If the solution is correct throughout, only then do you output -1.
4. The first verifier may be wrong. You cannot just accept their result. Always verify carefully and do not hesitate to disagree.

Your output format:
1. Begin with an overview: "Let's check the solution paragraph by paragraph based on the first agent's verification:"
2. For each paragraph (starting from paragraph 0), specify whether it is correct or not based on the first verifier's reason. If an error is found, explain the reason for the error. If correct, explain why it is correct.
3. End with the earliest error index in \\boxed{}, or -1 in \\boxed{} if no errors.
"""

    def prepare_prompts(self, entries: List[Dict], current_round: int) -> List[str]:
        all_prompts = []
        for entry in entries:
            problem = entry["problem"]
            tagged_response = process_steps(entry["steps"])
            
            for i in range(5):
                if current_round == 1:
                    prev_label_key = f"generated_label_{i}"
                    prev_reason_key = f"reason_{i}"
                else:
                    prev_label_key = f"round_{current_round-1}_label_{i}"
                    prev_reason_key = f"round_{current_round-1}_reason_{i}"
                
                prev_label = entry.get(prev_label_key)
                prev_reason = entry.get(prev_reason_key, "No previous reason")
                
                user_message = f"""
[Problem] 

{problem}

[Steps]

{tagged_response}

First verifier's results:
Generated Label: {prev_label}
Reason: {prev_reason}

Please provide your verification.
Analyze and return earliest error index in \\boxed{{}}, or -1 if no errors."""
                
                messages = [
                    {"role": "system", "content": self.verifier_system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                all_prompts.append(prompt)
        
        return all_prompts

    def generate_responses(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

def process_responses(entries: List[Dict], responses: List[str], current_round: int) -> List[Dict]:
    processed_entries = []
    for entry_idx, entry in enumerate(entries):
        current_entry = entry.copy()
        start_idx = entry_idx * 5
        current_responses = responses[start_idx:start_idx + 5]
        
        labels = []
        for i, response in enumerate(current_responses):
            current_entry[f"round_{current_round}_reason_{i}"] = response
            label = extract_answer(response)
            current_entry[f"round_{current_round}_label_{i}"] = label
            labels.append(label)
        valid_labels = [l for l in labels if l is not None]
        if valid_labels:
            majority_label = Counter(valid_labels).most_common(1)[0][0]
            majority_count = Counter(valid_labels).most_common(1)[0][1]
            proportion = majority_count / 5
        else:
            majority_label = None
            proportion = 0.0

        current_entry[f"round_{current_round}_majority_label"] = majority_label
        current_entry[f"round_{current_round}_majority_proportion"] = proportion
        processed_entries.append(current_entry)
    
    return processed_entries

def check_stopping_conditions(entries: List[Dict], current_round: int) -> List[Dict]:
    for entry in entries:
        if current_round >= 3:
            prev_labels = [entry[f"round_{r}_majority_label"] for r in range(current_round-2, current_round+1)]
            prev_proportions = [entry[f"round_{r}_majority_proportion"] for r in range(current_round-2, current_round+1)]
            
            if (len(set(prev_labels)) == 1 and 
                prev_proportions[0] <= prev_proportions[1] <= prev_proportions[2]):
                entry["final_label"] = prev_labels[-1]
                entry["final_reason"] = f"Stopped at round {current_round} with consistent majority"
                entry["finished"] = True
            else:
                entry["finished"] = False
        else:
            entry["finished"] = False
            
        if current_round >= 10:
            entry["final_label"] = entry[f"round_{current_round}_majority_label"]
            entry["final_reason"] = "Reached maximum rounds"
            entry["finished"] = True
            
    return entries

def process_dataset(dataset_name: str, vllm_client: VLLMInference, input_dir: str, output_dir: str):
    input_file = os.path.join(input_dir, f"mv_verified_results_{vllm_client.model_name}_{dataset_name}.jsonl")
    output_file = os.path.join(output_dir, f"inf_verified_results_{vllm_client.model_name}_{dataset_name}.jsonl")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f]
    
    os.makedirs(output_dir, exist_ok=True)
    out_f = open(output_file, 'w', encoding='utf-8')
    
    batch_size = 200
    current_round = 1
    active_entries = all_entries
    
    try:
        while active_entries and current_round <= 15:
            print(f"\nStarting round {current_round} with {len(active_entries)} active entries for {dataset_name}")
            
            all_responses = []
            with tqdm(total=len(active_entries), desc=f"Round {current_round}") as pbar:
                for i in range(0, len(active_entries), batch_size):
                    batch = active_entries[i:min(i + batch_size, len(active_entries))]
                    prompts = vllm_client.prepare_prompts(batch, current_round)
                    responses = vllm_client.generate_responses(prompts)
                    all_responses.extend(responses)
                    pbar.update(len(batch))
            
            processed_entries = process_responses(active_entries, all_responses, current_round)
            checked_entries = check_stopping_conditions(processed_entries, current_round)
            
            finished_entries = [entry for entry in checked_entries if entry.get("finished", False)]
            for entry in finished_entries:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
            
            active_entries = [entry for entry in checked_entries if not entry.get("finished", False)]
            current_round += 1
        
        for entry in active_entries:
            if not entry.get("finished", False):
                entry["final_label"] = entry.get(f"round_{current_round-1}_majority_label")
                entry["final_reason"] = "Reached maximum rounds without convergence"
                entry["finished"] = True
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    finally:
        out_f.close()
    
    print(f"Finished processing {dataset_name} dataset")

def main():
    parser = argparse.ArgumentParser(description='Process datasets with a specific model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output_dic', type=str, required=True, help='Directory that contains the previous output and where final results will be stored')
    args = parser.parse_args()
    
    datasets = ['gsm8k', 'math', 'olym', 'omni', 'prm', 'mathcheck']
    input_dir = args.output_dic
    output_dir = args.output_dic

    try:
        vllm_client = VLLMInference(args.model_path)
        for dataset in datasets:
            try:
                process_dataset(dataset, vllm_client, input_dir, output_dir)
            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")
                continue
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    except Exception as e:
        print(f"Error initializing model {args.model_path}: {e}")
    
    finally:
        if 'vllm_client' in locals():
            if hasattr(vllm_client, 'llm'):
                del vllm_client.llm
            if hasattr(vllm_client, 'tokenizer'):
                del vllm_client.tokenizer
            del vllm_client
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
