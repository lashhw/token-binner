import os
import json
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")

    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--num_proc', type=int, default=os.cpu_count() // 2)
    parser.add_argument('--output_file', type=str, default="indices.json")

    return parser.parse_args()


def get_token_length_category(length: int) -> str:
    if length < 1:
        assert False
    elif length <= 4096:
        return "<=4k"
    elif length <= 8192:
        return "4-8k"
    elif length <= 16384:
        return "8-16k"
    elif length <= 32768:
        return "16-32k"
    elif length <= 65536:
        return "32-64k"
    elif length <= 131072:
        return "64-128k"
    else:
        return ">128k"


def classify_by_token_length(batch, tokenizer):
    texts = []

    for messages in batch['messages']:
        assert len(messages) == 2

        user_message = messages[0]
        assistant_message = messages[1]

        assert set(user_message.keys()) == {'role', 'content'}
        assert set(assistant_message.keys()) == {'role', 'content'}

        assert user_message['role'] == 'user'
        assert assistant_message['role'] == 'assistant'

        question_text = user_message['content']
        answer_text = assistant_message['content']
        text = question_text + answer_text

        texts.append(text)

    tokenized_texts = tokenizer(texts)
    lengths = [len(ids) for ids in tokenized_texts['input_ids']]
    return {'token_length_category': [get_token_length_category(l) for l in lengths]}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("THUDM/LongAlign-10k", split="train")
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))

    processing_function = lambda batch: classify_by_token_length(batch, tokenizer)
    classified_dataset = dataset.map(
        processing_function,
        batched=True,
        num_proc=args.num_proc
    )

    category_indices = defaultdict(list)
    
    for i, category_info in enumerate(classified_dataset.select_columns(['token_length_category'])):
        category_name = category_info['token_length_category']
        category_indices[category_name].append(i)

    for category, indices in category_indices.items():
        print(f"{category}: {len(indices):,}")

    with open(args.output_file, 'w') as f:
        json.dump(category_indices, f)


if __name__ == '__main__':
    args = get_args()
    main(args)
