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
    parser.add_argument('--dataset_splits', type=str, default="high_part00,high_part01,high_part02")
    parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_args()


def get_token_length_category(length: int) -> str:
    if length < 1:
        assert False
    for upper_bound, label in TOKEN_BINS:
        if length < upper_bound:
            return label
    return DEFAULT_CATEGORY


def build_chat_messages(example):
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
    return [
        {"role": user_msg["role"], "content": user_msg["content"]},
        {"role": assistant_msg["role"], "content": assistant_content},
    ]


def update_progress(progress, category_counts, error_count):
    progress.set_postfix({
        **{category: category_counts[category] for category in ALL_CATEGORIES},
        "error": error_count,
    })


def process_batch(batch, tokenizer, category_indices, category_seen_counts, rng):
    chat_messages_list = [chat_messages for _, _, chat_messages in batch]
    texts = [
        tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        for chat_messages in chat_messages_list
    ]
    tokenized = tokenizer(texts, add_special_tokens=False)
    lengths = [len(ids) for ids in tokenized["input_ids"]]

    for (row_id, example, _), token_length in zip(batch, lengths):
        category_name = get_token_length_category(token_length)
        category_seen_counts[category_name] += 1

        if category_name in OUTPUT_FILES:
            example["original_row_id"] = row_id
            example["token_length"] = token_length

            target_count = TARGET_COUNTS[category_name]
            reservoir = category_indices[category_name]

            if len(reservoir) < target_count:
                reservoir.append(example)
            else:
                j = rng.randint(0, category_seen_counts[category_name] - 1)
                if j < target_count:
                    reservoir[j] = example


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset_splits = args.dataset_splits.split(",")

    category_indices = defaultdict(list)
    category_seen_counts = defaultdict(int)
    error_count = 0
    rng = random.Random(42)

    for dataset_split in dataset_splits:
        dataset = load_dataset("nvidia/Nemotron-Math-v2", split=dataset_split, streaming=True)
        batch = []
        seen_count = 0

        progress = tqdm(enumerate(dataset), desc=dataset_split, mininterval=1.0)
        for i, example in progress:
            seen_count += 1
            chat_messages = build_chat_messages(example)
            if chat_messages is None:
                error_count += 1
                update_progress(progress, category_seen_counts, error_count)
                continue

            batch.append((i, example, chat_messages))
            if len(batch) >= args.batch_size:
                process_batch(batch, tokenizer, category_indices, category_seen_counts, rng)
                batch = []

            update_progress(progress, category_seen_counts, error_count)

        if batch:
            process_batch(batch, tokenizer, category_indices, category_seen_counts, rng)
            update_progress(progress, category_seen_counts, error_count)

    for category, output_path in OUTPUT_FILES.items():
        with open(output_path, "w", encoding="utf-8") as output_file:
            for sample in category_indices[category]:
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    args = get_args()
    main(args)
