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
TOKEN_BINS = [
    (4096, "<4k"),
    (8192, "4-8k"),
    (16384, "8-16k"),
    (32768, "16-32k"),
    (65536, "32-64k"),
    (131072, "64-128k"),
]
DEFAULT_CATEGORY = ">=128k"
ALL_CATEGORIES = [label for _, label in TOKEN_BINS] + [DEFAULT_CATEGORY]


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument('--dataset_subset', type=str, default="medium")

    return parser.parse_args()


def get_token_length_category(length: int) -> str:
    if length < 1:
        assert False
    for upper_bound, label in TOKEN_BINS:
        if length < upper_bound:
            return label
    return DEFAULT_CATEGORY


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

    assistant_content = f"<think>{assistant_msg['reasoning_content']}</think>{assistant_msg['content']}"
    chat_messages = [
        {"role": user_msg["role"], "content": user_msg["content"]},
        {"role": assistant_msg["role"], "content": assistant_content},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_length = len(input_ids)
    return get_token_length_category(token_length), token_length


def update_progress(progress, error_count, category_counts):
    progress.set_postfix({
        "errors": error_count,
        **{category: category_counts[category] for category in ALL_CATEGORIES},
    })


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("nvidia/Nemotron-Math-v2", split=args.dataset_subset, streaming=True)

    category_indices = defaultdict(list)
    category_seen_counts = defaultdict(int)
    less_than_4k_count = 0
    error_count = 0
    rng = random.Random(42)

    progress = tqdm(enumerate(dataset))
    for i, example in progress:
        category_info = classify_by_token_length(example, tokenizer)
        if category_info is None:
            error_count += 1
            update_progress(progress, error_count, category_indices)
            continue

        category_name, token_length = category_info
        category_seen_counts[category_name] += 1

        if category_name == '<4k':
            less_than_4k_count += 1
        elif category_name in OUTPUT_FILES:
            example["original_row_id"] = i
            example["token_length"] = token_length

            target_count = TARGET_COUNTS[category_name]
            reservoir = category_indices[category_name]

            if len(reservoir) < target_count:
                reservoir.append(example)
            else:
                j = rng.randint(0, category_seen_counts[category_name] - 1)
                if j < target_count:
                    reservoir[j] = example

        update_progress(progress, error_count, category_seen_counts)

    for category in OUTPUT_FILES:
        output_path = OUTPUT_FILES[category]
        with open(output_path, "w", encoding="utf-8") as output_file:
            for sample in category_indices[category]:
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"<4k: {less_than_4k_count:,}")
    for category, samples in category_indices.items():
        print(f"{category}: {len(samples):,}")


if __name__ == '__main__':
    args = get_args()
    main(args)
