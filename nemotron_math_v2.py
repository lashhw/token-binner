import os
import json
import argparse
import random
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

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

    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument('--dataset_subset', type=str, default="medium")

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
    except:
        return None

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
    return {
        'token_length_category': get_token_length_category(token_length),
        'token_length': token_length,
    }


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("nvidia/Nemotron-Math-v2", split=args.dataset_subset, streaming=True)

    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = {
        category: os.path.join(args.output_dir, filename)
        for category, filename in OUTPUT_FILES.items()
    }
    category_indices = defaultdict(list)
    category_seen_counts = defaultdict(int)
    less_than_4k_count = 0
    error_count = 0
    rng = random.Random(42)

    iterator = enumerate(dataset)
    total = args.num_samples if args.num_samples > 0 else None
    progress = tqdm(iterator, total=total)
    for i, example in progress:
        category_info = classify_by_token_length(example, tokenizer)
        if category_info is None:
            error_count += 1
            progress.set_postfix(errors=error_count)
            if args.num_samples > 0 and i + 1 >= args.num_samples:
                break
            continue

        category_name = category_info['token_length_category']
        token_length = category_info['token_length']
        if category_name == '<4k':
            less_than_4k_count += 1
            if args.num_samples > 0 and i + 1 >= args.num_samples:
                break
            continue

        if category_name not in OUTPUT_FILES:
            if args.num_samples > 0 and i + 1 >= args.num_samples:
                break
            continue

        example["original_row_id"] = i
        example["token_length"] = token_length
        example["token_length_category"] = category_name
        category_seen_counts[category_name] += 1
        target_count = TARGET_COUNTS[category_name]
        reservoir = category_indices[category_name]
        if len(reservoir) < target_count:
            reservoir.append(example)
        else:
            j = rng.randint(0, category_seen_counts[category_name] - 1)
            if j < target_count:
                reservoir[j] = example

        if args.num_samples > 0 and i + 1 >= args.num_samples:
            break

    for category in OUTPUT_FILES:
        output_path = output_paths[category]
        with open(output_path, "w", encoding="utf-8") as output_file:
            for sample in category_indices[category]:
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"<4k: {less_than_4k_count:,}")
    for category, samples in category_indices.items():
        print(f"{category}: {len(samples):,}")


if __name__ == '__main__':
    args = get_args()
    main(args)
