# Online Language Modelling Training Pipeline

This repo has the code for training models and tokenizers on the olm data, but it should work with any Hugging Face dataset with text examples. You can see the models for the OLM project here: https://huggingface.co/olm. They actually get better performance than their original static counterparts.

## Creating a Tokenizer and Tokenizing Datasets

If you want to tokenize and prepare the datasets from scratch you can adapt the arguments below  

```bash
python create_tokenizer.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 Tristan/olm-CC-MAIN-2022-21-sampling-ratio-0.14775510204 Tristan/olm-CC-MAIN-2022-27-sampling-ratio-0.16142697881 Tristan/olm-CC-MAIN-2022-33-sampling-ratio-0.20 --existing_tokenizer_template roberta-base --output_tokenizer_name Tristan/olm-tokenizer --text_column text --push_to_hub
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-tokenizer --output_dataset_name Tristan/olm-october-2022-tokenized-512 --text_column text --num_proc 224 --push_to_hub --max_len 512
python chunk_and_tokenize_datasets.py --input_dataset_names Tristan/olm-wikipedia-20221001 Tristan/olm-CC-MAIN-2022-40-sampling-ratio-0.15894621295 --input_tokenizer_name Tristan/olm-tokenizer --output_dataset_name Tristan/olm-october-2022-tokenized-1024 --text_column text --num_proc 224 --push_to_hub --max_len 1024
```

But you can also use already trained tokinizer and latest tokenized datasets which were already pre-computed here: https://huggingface.co/olm

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_model.py --lm_type=mlm --dataset_id=olm/olm-december-2022-tokenized-512 --repository_id=muhtasham/olm-bert-tiny-december-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=google/bert_uncased_L-2_H-128_A-2 --adam_beta2=0.98 --adam_epsilon=1e-6 --adam_beta1=0.9 --warmup_steps=24000 --max_steps=100000 --per_device_train_batch_size=20 --gradient_accumulation_steps=25 --learning_rate=6e-4
```

## Training a BERT/RoBERTa model from scratch on 410B tokens (this is the 100k step option in the RoBERTa paper, which uses about the same compute as the original BERT used)

Below script trains a more up-to-date version of the original BERT and original RoBERTa. 

```bash
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=mlm --dataset_id=Tristan/olm-october-2022-tokenized-512 --repository_id=Tristan/olm-roberta-base-oct-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=roberta-base --adam_beta2=0.98 --adam_epsilon=1e-6 --adam_beta1=0.9 --warmup_steps=24000 --max_steps=100000 --per_device_train_batch_size=20 --gradient_accumulation_steps=25 --learning_rate=6e-4
```

## Training a GPT2 model from scratch on 300B tokens (the number of tokens reported in the GTP3 paper)

Below script trains a more up-to-date version of the original GPT-2. 

```bash
python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=clm --dataset_id=Tristan/olm-october-2022-tokenized-1024 --repository_id=Tristan/olm-gpt2-oct-2022 --tokenizer_id=Tristan/olm-tokenizer --model_config_id=gpt2 --max_steps=580000 --learning_rate=1e-3 --warmup_steps=725 --adam_beta1=0.9 --adam_beta2=0.95 --adam_epsilon=1e-7 --weight_decay=0.1 --lr_scheduler_type=cosine --per_device_train_batch_size=8 --gradient_accumulation_steps=4
```

### Details on compute
Pretraining takes quite a bit of compute. Both olm-gpt2 and olm-roberta-base each used 16 40GB A100's with around 1 TB of disk space (although could be less) running for about 5-6 days. 
