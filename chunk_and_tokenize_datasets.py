from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from itertools import chain
import math
import argparse

parser = argparse.ArgumentParser(description="Constructs a tokenized dataset from an input tokenizer and a list of input datasets. Also chunks the dataset into examples of max_len tokens. This script guarantees that the fraction of examples which are padded to reach max_len tokens is <= 1/1000. The other examples will be max_len without padding. This amount of padding shouldn't really affect dataset size or training speed.")
parser.add_argument("--input_dataset_names", nargs='+', required=True)
parser.add_argument("--input_tokenizer_name", required=True)
parser.add_argument("--output_dataset_name", required=True)
parser.add_argument("--text_column", required=True)
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--max_len", type=int, help="Max length for the tokenizer.", default=512)
parser.add_argument("--num_proc", type=int, help="The number of processes to use.", required=True)
args = parser.parse_args()

ds_list = []
for dataset_name in args.input_dataset_names:
    ds = load_dataset(dataset_name, split="train")
    ds = ds.remove_columns([col for col in ds.column_names if col != args.text_column])  # only keep the text column
    ds_list.append(ds)

raw_ds = concatenate_datasets(ds_list)

tokenizer = AutoTokenizer.from_pretrained(args.input_tokenizer_name)

def tokenize(example):
    tokenized_example = tokenizer(
       example[args.text_column], return_special_tokens_mask=True
    )
    return tokenized_example

# tokenize dataset
tokenized_ds = raw_ds.map(tokenize, remove_columns=[args.text_column], num_proc=args.num_proc)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We add a little padding so these tokens can be evenly split into examples with max_len # of tokens.
    if total_length >= args.max_len:
        remainder  = total_length - (total_length // args.max_len) * args.max_len
        if remainder > 0:
            concatenated_examples["input_ids"] += [tokenizer.pad_token_id]*(args.max_len - remainder)
            concatenated_examples["special_tokens_mask"] += [1]*(args.max_len - remainder)
            concatenated_examples["attention_mask"] += [0]*(args.max_len - remainder)
            if "token_type_ids" in concatenated_examples:
                # token_type_ids is 0 - we don't support next-sentence-prediction.
                concatenated_examples["token_type_ids"] += [0]*(args.max_len - remainder)
            total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + args.max_len] for i in range(0, total_length, args.max_len)]
        for k, t in concatenated_examples.items()
    }
    return result

# Note that because the batch size is 1000, the fraction of examples with pad tokens will only be <= 1/1000.
# The rest of the examples will have a full max_len tokens without padding.
tokenized_ds = tokenized_ds.map(group_texts, batched=True, batch_size=1000, num_proc=args.num_proc)

print(f"the dataset contains in total {len(tokenized_ds)*args.max_len} tokens")

tokenized_ds.save_to_disk(args.output_dataset_name)

if args.push_to_hub:
    tokenized_ds.push_to_hub(args.output_dataset_name)

