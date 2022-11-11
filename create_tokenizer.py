from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from itertools import chain
import math
import argparse

parser = argparse.ArgumentParser(description="Constructs a Tokenizer from a list of input datasets. It uses all available CPUs by default.")
parser.add_argument("--input_dataset_names", nargs='+', required=True, help="The datasets to use. They will be concatenated if there is more than one.")
parser.add_argument("--existing_tokenizer_template", required=True)
parser.add_argument("--output_tokenizer_name", required=True)
parser.add_argument("--text_column", required=True)
parser.add_argument("--vocab_size", default=None, type=int)
parser.add_argument("--push_to_hub", action="store_true")
args = parser.parse_args()

ds_list = []
for dataset_name in args.input_dataset_names:
    ds = load_dataset(dataset_name, split="train")
    ds = ds.remove_columns([col for col in ds.column_names if col != args.text_column])  # only keep the text column
    ds_list.append(ds)

raw_ds = concatenate_datasets(ds_list)

# create a python generator to dynamically load the data
def batch_iterator(batch_size=400000):
    for i in tqdm(range(0, len(raw_ds), batch_size)):
        yield raw_ds[i : i + batch_size][args.text_column]

# create a tokenizer from existing one to re-use special tokens and vocab size.
tokenizer = AutoTokenizer.from_pretrained(args.existing_tokenizer_template)

tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=tokenizer.vocab_size if args.vocab_size is None else args.vocab_size)

tokenizer.save_pretrained(args.output_tokenizer_name)

if args.push_to_hub:
    tokenizer.push_to_hub(args.output_tokenizer_name)

