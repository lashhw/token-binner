import os
import json
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--dataset_name', type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument('--dataset_subset', type=str, default="sample-10BT")

    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--num_proc', type=int, default=os.cpu_count() // 2)
    parser.add_argument('--output_file', type=str, default="indices.json")

    return parser.parse_args()

def get_token_length_category(length: int) -> str:
    if 1 <= length < 4096:
        return "<4k"
    elif 4096 <= length < 8192:
        return "4-8k"
    elif 8192 <= length < 16384:
        return "8-16k"
    elif 16384 <= length < 32768:
        return "16-32k"
    elif 32768 <= length < 65536:
        return "32-64k"
    elif 65536 <= length < 131072:
        return "64-128k"
    elif length >= 131072:
        return ">=128k"
    else:
        assert False

def classify_by_token_length(batch, tokenizer):
    tokenized_texts = tokenizer(batch['text'])
    lengths = [len(ids) for ids in tokenized_texts['input_ids']]
    batch['token_length_category'] = [get_token_length_category(l) for l in lengths]
    return batch

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(args.dataset_name, name=args.dataset_subset, split="train")
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))

    processing_function = lambda batch: classify_by_token_length(batch, tokenizer)
    
    classified_dataset = dataset.map(
        processing_function,
        batched=True,
        num_proc=args.num_proc
    )

    category_indices = defaultdict(list)
    less_than_4k_count = 0
    
    for i, category_info in enumerate(classified_dataset.select_columns(['token_length_category'])):
        category_name = category_info['token_length_category']
        if category_name == '<4k':
            less_than_4k_count += 1
        else:
            category_indices[category_name].append(i)

    print(f"<4k: {less_than_4k_count:,}")
    for category, indices in category_indices.items():
        print(f"{category}: {len(indices):,}")

    with open(args.output_file, 'w') as f:
        json.dump(category_indices, f)

if __name__ == '__main__':
    args = get_args()
    main(args)
