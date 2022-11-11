# Online Language Modelling Training Pipeline

This repo has the code for training models and tokenizers on the olm data, but it should work with any Hugging Face dataset with text examples.

## Training a BERT model from scratch with the MLM pretraining objective

```
python create_tokenizer.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 Tristan/olm-CC-MAIN-2022-21-sampling-ratio-0.14775510204 Tristan/olm-CC-MAIN-2022-27-sampling-ratio-0.16142697881 Tristan/olm-CC-MAIN-2022-33-sampling-ratio-0.20 --existing_tokenizer_template bert-base-uncased --output_tokenizer_name Tristan/olm-bert-base-uncased --text_column text --vocab_size 50000 --push_to_hub
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-bert-base-uncased --output_dataset_name Tristan/olm-october-2022-tokenized-olm-bert-base-uncased --text_column text --num_proc 224 --push_to_hub
python -m torch.distributed.launch --nproc_per_node=16 run_mlm.py --dataset_id=Tristan/olm-october-2022-tokenized-olm-bert-base-uncased --repository_id=Tristan/olm-bert-base-uncased-oct-2022 --tokenizer_id=Tristan/olm-bert-base-uncased
```

## Training a BLOOM model from scratch with the CLM pretraining objective

```
python create_tokenizer.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 Tristan/olm-CC-MAIN-2022-21-sampling-ratio-0.14775510204 Tristan/olm-CC-MAIN-2022-27-sampling-ratio-0.16142697881 Tristan/olm-CC-MAIN-2022-33-sampling-ratio-0.20 --existing_tokenizer_template bloom --output_tokenizer_name Tristan/olm-bert-base-uncased --text_column text --vocab_size 50000 --push_to_hub
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-bloom --output_dataset_name Tristan/olm-october-2022-tokenized-olm-bloom --text_column text --num_proc 224 --push_to_hub
python -m torch.distributed.launch --nproc_per_node=16 run_clm.py --dataset_id=Tristan/olm-october-2022-tokenized-olm-bloom --repository_id=Tristan/olm-bloom-oct-2022 --tokenizer_id=Tristan/olm-bloom
```

