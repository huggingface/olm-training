from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from itertools import chain
import argparse

parser = argparse.ArgumentParser(description="Constructs a Tokenizer and Tokenized dataset from a list of input datasets.")
parser.add_argument("--input_dataset_names", nargs='+', required=True)
parser.add_argument("--output_dataset_name", required=True)
parser.add_argument("--output_tokenizer_name", required=True)
parser.add_argument("--num_proc", type=int, help="The number of processes to use.", required=True)
args = parser.parse_args()

ds_list = []
for dataset_name in args.input_dataset_names:
    ds = load_dataset(dataset_name, split="train")
    ds = ds.remove_columns([col for col in ds.column_names if col != "text"])  # only keep the 'text' column
    ds_list.append(ds)

raw_ds = concatenate_datasets(ds_list)

# repository id for saving the tokenizer
tokenizer_id="olm_roberta_" + "_".join([dataset_name for dataset_name in args.input_dataset_names])

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(raw_ds), batch_size)):
        yield raw_ds[i : i + batch_size]["text"]

# create a tokenizer from existing one to re-use special tokens
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

roberta_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=50265)

def tokenize(example):
    tokenized_example = tokenizer(
       example["text"], return_special_tokens_mask=True
    )
    return tokenized_example

# preprocess dataset
tokenized_ds = raw_ds.map(lambda example: {"tokens": tokenize(example)}, remove_columns=["text"], num_proc=args.num_proc)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
# TODO
for example in tokenized_ds:
    print(example)
    crash


# shuffle dataset
tokenized_ds = tokenized_ds.shuffle(seed=42)

print(f"the dataset contains in total {len(tokenized_ds)*tokenizer.model_max_length} tokens")

tokenized_ds.save_to_disk(args.output_dataset_name)
tokenizer.save_to_disk(args.output_tokenizer_name)

if args.push_to_hub:
    tokenized_ds.push_to_hub(args.output_dataset_name)
    tokenizer.push_to_hub(args.output_tokenizer_name)

