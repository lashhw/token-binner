import os
import json
import argparse
import random
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

DATASET_NAME = "nvidia/Nemotron-Math-v2"
SPLIT = "medium"

OUTPUT_FILES = {
    "4-8k": "nemotron_math_v2_4k.jsonl",
    "8-16k": "nemotron_math_v2_8k.jsonl",
    "16-32k": "nemotron_math_v2_16k.jsonl",
    "32-64k": "nemotron_math_v2_32k.jsonl",
}
TARGET_COUNTS = {
    "4-8k": 4000,
    "8-16k": 2000,
    "16-32k": 1000,
    "32-64k": 500,
}


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME)
    parser.add_argument('--split', type=str, default=SPLIT)

    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=".")

    return parser.parse_args()


def get_token_length_category(length: int) -> str:
    if length < 1:
        assert False
    elif length < 4096:
        return "<4k"
    elif length < 8192:
        return "4-8k"
    elif length < 16384:
        return "8-16k"
    elif length < 32768:
        return "16-32k"
    elif length < 65536:
        return "32-64k"
    elif length < 131072:
        return "64-128k"
    else:
        return ">=128k"


def classify_by_token_length(example, tokenizer):
    try:
        messages = example["messages"]
        assert isinstance(messages, list) and len(messages) == 2

        user_msg = messages[0]
        assistant_msg = messages[1]
        assert isinstance(user_msg, dict)
        assert isinstance(assistant_msg, dict)

        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert user_msg["reasoning_content"] is None

        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], str)
        assert isinstance(assistant_msg["reasoning_content"], str)
    except (AssertionError, KeyError, TypeError):
        return None, None

    chat_messages = [
        {"role": user_msg["role"], "content": user_msg["content"]},
        {"role": assistant_msg["role"], "content": assistant_msg["content"]},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_length = len(input_ids)
    return get_token_length_category(token_length), token_length


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = {
        category: os.path.join(args.output_dir, filename)
        for category, filename in OUTPUT_FILES.items()
    }
    reservoirs = {category: [] for category in OUTPUT_FILES}
    seen_counts = defaultdict(int)
    error_count = 0
    rng = random.Random(42)

    try:
        iterator = enumerate(dataset)
        total = args.num_samples if args.num_samples > 0 else None
        progress = tqdm(iterator, total=total)
        for i, example in progress:
            category, token_length = classify_by_token_length(example, tokenizer)
            if category is None:
                error_count += 1
                progress.set_postfix(errors=error_count)
                if args.num_samples > 0 and i + 1 >= args.num_samples:
                    break
                continue

            if category not in OUTPUT_FILES:
                if args.num_samples > 0 and i + 1 >= args.num_samples:
                    break
                continue

            example["original_row_id"] = i
            example["token_length"] = token_length
            example["token_length_category"] = category
            seen_counts[category] += 1
            target_count = TARGET_COUNTS[category]
            reservoir = reservoirs[category]
            if len(reservoir) < target_count:
                reservoir.append(example)
            else:
                j = rng.randint(0, seen_counts[category] - 1)
                if j < target_count:
                    reservoir[j] = example

            if args.num_samples > 0 and i + 1 >= args.num_samples:
                break
    finally:
        pass

    for category, samples in reservoirs.items():
        output_path = output_paths[category]
        with open(output_path, "w", encoding="utf-8") as output_file:
            for sample in samples:
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    for category, samples in reservoirs.items():
        print(f"{category}: {len(samples):,}")
    for category, path in output_paths.items():
        print(f"Wrote {category} to {path}")


if __name__ == '__main__':
    args = get_args()
    main(args)
