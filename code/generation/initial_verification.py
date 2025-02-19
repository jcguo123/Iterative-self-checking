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

# Set CUDA memory allocator settings
torch.cuda.memory._set_allocator_settings('expandable_segments:False')

def process_steps(steps):
    tagged_response = ""
    for i, step in enumerate(steps):
        tagged_response += f'<paragraph_{i}>\n{step}\n</paragraph_{i}>\n\n'
    return tagged_response.strip()

def extract_answer(solution_text: str):
    if not solution_text:
        return None
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

class VLLMInference:
    def __init__(self, model_path: str):
        # Use the basename of the model path as the model name
        self.model_name = os.path.basename(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            gpu_memory_utilization=0.90,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_prefix_caching=True,
            swap_space=16,
            max_num_seqs=200
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_tokens=8192,
            seed=42
        )

    def prepare_prompt(self, problem: str, tagged_response: str) -> List[Dict]:
        content = f"""The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}."""
        messages = [{"role": "user", "content": content}]
        return messages

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        all_messages = []
        for entry in batch:
            tagged_response = process_steps(entry["steps"])
            messages = self.prepare_prompt(entry["problem"], tagged_response)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Generate 5 responses per entry
            all_messages.extend([prompt] * 5)

        outputs = self.llm.generate(all_messages, self.sampling_params)

        processed_entries = []
        for i in range(0, len(outputs), 5):
            entry = batch[i // 5].copy()
            labels = []
            for j in range(5):
                response = outputs[i + j].outputs[0].text
                entry[f"reason_{j}"] = response
                label = extract_answer(response)
                entry[f"generated_label_{j}"] = label
                labels.append(label)
            majority_label = Counter(labels).most_common(1)[0][0]
            entry["majority_generated_label"] = majority_label
            processed_entries.append(entry)

        return processed_entries

def process_dataset(dataset_name: str, vllm_client: VLLMInference, input_dir: str, output_dir: str):
    input_file = os.path.join(input_dir, f"{dataset_name}.jsonl")
    output_file = os.path.join(output_dir, f"mv_verified_results_{vllm_client.model_name}_{dataset_name}.jsonl")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    batch_size = 200
    processed_entries = []
    
    with tqdm(total=len(entries), desc=f"Processing {dataset_name} with {vllm_client.model_name}") as pbar:
        for i in range(0, len(entries), batch_size):
            batch = entries[i:min(i + batch_size, len(entries))]
            batch_results = vllm_client.process_batch(batch)
            processed_entries.extend(batch_results)
            pbar.update(len(batch))

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in processed_entries:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Process datasets with a specific model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Directory containing the dataset JSONL files')
    parser.add_argument('--output_dic', type=str, required=True,
                        help='Directory to save the output results')
    args = parser.parse_args()
    
    datasets = ['gsm8k', 'math', 'olym', 'omni', 'mathcheck', 'prm']
    input_dir = args.dataset_path
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
