# Online Language Modelling Training Pipeline

This repo has the code for training models and tokenizers on the olm data, but it should work with any Hugging Face dataset with text examples. You can see the models for the OLM project here: https://huggingface.co/olm. They actually get better performance than their original static counterparts.

## Creating a Tokenizer and Tokenizing Datasets 

Here is an example of how to tokenize the datasets and train a tokenizer:

```bash
python create_tokenizer.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 Tristan/olm-CC-MAIN-2022-21-sampling-ratio-0.14775510204 Tristan/olm-CC-MAIN-2022-27-sampling-ratio-0.16142697881 Tristan/olm-CC-MAIN-2022-33-sampling-ratio-0.20 --existing_tokenizer_template roberta-base --output_tokenizer_name Tristan/olm-tokenizer --text_column text --push_to_hub
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-tokenizer --output_dataset_name Tristan/olm-october-2022-tokenized-512 --text_column text --num_proc 224 --push_to_hub --max_len 512
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-tokenizer --output_dataset_name Tristan/olm-october-2022-tokenized-1024 --text_column text --num_proc 224 --push_to_hub --max_len 1024
```

If you just want to train a model on the existing OLM data, you may be able to skip this step, though. We already have a trained tokenizer and tokenized datasets [here](https://huggingface.co/olm)

## Training a BERT/RoBERTa model from scratch on 410B tokens (this is the 100k step option in the RoBERTa paper, which uses about the same compute as the original BERT used)

```bash
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=mlm --dataset_id=Tristan/olm-october-2022-tokenized-512 --repository_id=Tristan/olm-roberta-base-oct-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=roberta-base --adam_beta2=0.98 --adam_epsilon=1e-6 --adam_beta1=0.9 --warmup_steps=24000 --max_steps=100000 --per_device_train_batch_size=20 --gradient_accumulation_steps=25 --learning_rate=6e-4
```

Note that the best hyperparameters are sensitive to both model architecture and scale. We found these hyperparameters to work well for the `roberta-base` model, but they may not work as well for e.g. `roberta-large`, or another architecture entirely.

## Training a GPT2 model from scratch on 300B tokens (the number of tokens reported in the GTP3 paper)

```bash
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=clm --dataset_id=Tristan/olm-october-2022-tokenized-1024 --repository_id=Tristan/olm-gpt2-oct-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=gpt2 --max_steps=580000 --learning_rate=1e-3 --warmup_steps=725 --adam_beta1=0.9 --adam_beta2=0.95 --adam_epsilon=1e-7 --weight_decay=0.1 --lr_scheduler_type=cosine --per_device_train_batch_size=8 --gradient_accumulation_steps=4
```

Note that the best hyperparameters are sensitive to both model architecture and scale. We found these hyperparameters to work well for the `gpt2` model, but they may not work as well for e.g. `gpt2-large`, or another architecture entirely.

## Training a T5 model from scratch

Note that it is also possible to train T5, although we haven't tuned the hyperparameters and we aren't trainig the T5 ourselves for the OLM project. If you want to train T5, you would specify arguments like this (but please take the time to find good hyperparameters yourself!).

```bash
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=t5 --dataset_id=Tristan/olm-october-2022-tokenized-568 --repository_id=Tristan/olm-t5-small-oct-2022 --tokenizer_id=Tristan/olm-t5-tokenizer --model_config_id=t5-small --adam_beta2=0.98 --adam_epsilon=1e-6 --adam_beta1=0.9 --warmup_steps=24000 --max_steps=100000 --per_device_train_batch_size=20 --gradient_accumulation_steps=25 --learning_rate=6e-4
```

Also note:

1. If you want your T5 to have an input length of 512, you need to pass it a tokenized dataset with examples of length 568. This is because the T5 denoising pretraining objective turns several tokens into one token, so the 568 tokens will be turned into 512 tokens before they are passed into the model.
2. You should train a separate OLM tokenizer with the `create_tokenizer.py` script above, and it should be based on the T5 tokenizer template to ensure that the tokenizer has the special denoising characters (e.g., just make `--existing_tokenizer_template=t5-small`).

## DeepSpeed compatibility

Our `train_model.py` script is compatible with DeepSpeed, enabling you to train big models (which do not fit on a single GPU) accross a cluster of nodes. Just specify `--deepspeed=<path to your deepspeed config>` in the `train_model.py` arguments to use it. An example of a DeepSpeed config that you could use is [here](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero3-example)

## Details on compute
To train both our OLM GPT2 and OLM BERT/RoBERTa, we use a machine with 16 40GB A100's and around 1 TB of disk space. Each model takes about 5-6 days to train with this machine.
