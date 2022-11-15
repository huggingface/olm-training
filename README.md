# Online Language Modelling Training Pipeline (WIP)

This repo has the code for training models and tokenizers on the olm data, but it should work with any Hugging Face dataset with text examples.

## Creating a Tokenizer and Tokenizing Datasets

```
python create_tokenizer.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 Tristan/olm-CC-MAIN-2022-21-sampling-ratio-0.14775510204 Tristan/olm-CC-MAIN-2022-27-sampling-ratio-0.16142697881 Tristan/olm-CC-MAIN-2022-33-sampling-ratio-0.20 --existing_tokenizer_template roberta-base --output_tokenizer_name Tristan/olm-tokenizer --text_column text --push_to_hub
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-tokenizer --output_dataset_name Tristan/olm-october-2022-tokenized-512 --text_column text --num_proc 224 --push_to_hub --max_len 512
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-tokenizer --output_dataset_name Tristan/olm-october-2022-tokenized-1024 --text_column text --num_proc 224 --push_to_hub --max_len 1024
```

## Training a BERT model from scratch with the MLM pretraining objective

```
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=mlm --dataset_id=Tristan/olm-october-2022-tokenized-512 --repository_id=Tristan/olm-roberta-base-oct-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=roberta-base
```

## Training a BLOOM model from scratch with the CLM pretraining objective

```
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=clm --dataset_id=Tristan/olm-october-2022-tokenized-1024 --repository_id=Tristan/olm-bloom-560m-oct-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=bigscience/bloom-560m --max_steps=90000 --learning_rate=3e-4 --warmup_steps=3000 --adam_beta2=0.95 --weight_decay=0.1 --lr_scheduler_type=cosine
```

