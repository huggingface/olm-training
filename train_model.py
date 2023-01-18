import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from t5_data_collator import DataCollatorForT5MLM, compute_t5_input_and_target_lengths

from transformers import (
    HfArgumentParser,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    set_seed,
    AutoConfig,
    DataCollatorForLanguageModeling,
)

from transformers import Trainer, TrainingArguments
from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments which aren't included in the TrainingArguments
    """

    dataset_id: str = field(
        default=None, metadata={"help": "The repository id of the dataset to use (via the datasets library)."}
    )
    tokenizer_id: str = field(
        default=None, metadata={"help": "The repository id of the tokenizer to use (via AutoTokenizer)."}
    )
    repository_id: str = field(
        default=None,
        metadata={"help": "The repository id where the model will be saved or loaded from for futher pre-training."},
    )
    hf_hub_token: str = field(
        default=True,
        metadata={"help": "The Token used to push models, metrics and logs to the Hub."},
    )
    lm_type: str = field(
        default=None,
        metadata={"help": "The type of language model to train. Options are mlm, clm, or t5. t5 is a WIP for now and may not work as expected."},
    )
    model_config_id: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The Batch Size per HPU used during training"},
    )
    max_steps: Optional[int] = field(
        default=460000,
        metadata={"help": "The Number of Training steps to perform."},
    )
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "Learning Rate for the training"})
    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mean_noise_span_length: Optional[float] = field(
        default=3.0, metadata={"help": "Mean span length of masked tokens for T5."}
    )
    desired_t5_input_length: Optional[int] = field(
        default=512, metadata={"help": "The desired number of tokens for T5 input. Because T5 masking alters the number of tokens, this value must be specified."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "Number of gradient accumulation steps to take; artificially increases the batch size"}
    )
    warmup_steps: Optional[int] = field(
        default=10000, metadata={"help": "Number of learning warmup steps to take"}
    )
    adam_beta1: Optional[float] = field(
        default=0.9, metadata={"help": "Parameter for the adam optimizer"}
    )
    adam_beta2: Optional[float] = field(
        default=0.999, metadata={"help": "Parameter for the adam optimizer"}
    )
    adam_epsilon: Optional[float] = field(
        default=1e-6, metadata={"help": "Parameter for the adam optimizer"}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Parameter for the adam optimizer. Regularization to prevent weights from getting too big."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "LR scheduler type, such as cosine or linear."}
    )
    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config if using deepspeed"})


def train_model():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Script parameters {script_args}")

    # set seed for reproducibility
    seed = 34
    set_seed(seed)
    
    # define our hyperparameters
    training_args = TrainingArguments(
        output_dir=script_args.repository_id,
        local_rank=script_args.local_rank,
        deepspeed=script_args.deepspeed,
        # logging & evaluation strategies
        logging_dir=f"{script_args.repository_id}/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=script_args.repository_id,
        hub_token=script_args.hf_hub_token,
        # optimization parameters
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        learning_rate=script_args.learning_rate,
        seed=seed,
        max_steps=script_args.max_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        warmup_steps=script_args.warmup_steps,
        adam_beta1=script_args.adam_beta1,
        adam_beta2=script_args.adam_beta2,
        bf16 = True,
        adam_epsilon=script_args.adam_epsilon,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,
    )

    # load processed dataset
    train_dataset = load_dataset(script_args.dataset_id, split="train")

    # load trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id, use_auth_token=script_args.hf_hub_token)

    # load config (for training from scratch, we call .from_config())
    logger.info("Training new model from scratch")
    config = AutoConfig.from_pretrained(script_args.model_config_id)

    # Set up the model and data collator.
    if script_args.lm_type == "mlm":
        model = AutoModelForMaskedLM.from_config(config)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=script_args.mlm_probability, pad_to_multiple_of=8
        )
    elif script_args.lm_type == "clm":
        model = AutoModelForCausalLM.from_config(config)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )
    elif script_args.lm_type == "t5":
        train_dataset = train_dataset.remove_columns(["attention_mask", "special_tokens_mask"])
        input_length = script_args.desired_t5_input_length
        expanded_inputs_length, target_length = compute_t5_input_and_target_lengths(
            inputs_length=input_length,
            noise_density=script_args.mlm_probability,
            mean_noise_span_length=script_args.mean_noise_span_length,
        )
        assert expanded_inputs_length == len(train_dataset[0]["input_ids"]),\
            f"""
            You have specified that the T5 input length should be {script_args.desired_t5_input_length}.
            In order to do this, the examples in the dataset need to be {expanded_inputs_length} before masking.
            But the examples in the dataset actually appear to be {len(train_dataset[0]['input_ids'])} tokens long.
            """
        model = T5ForConditionalGeneration._from_config(config)
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=script_args.mlm_probability,
            mean_noise_span_length=script_args.mean_noise_span_length,
            input_length=input_length,
            target_length=target_length,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
    )
    else:
        raise ValueError("Unrecognized lm_type. Options are mlm, clm, or t5.")

    logger.info(f"Resizing token embedding to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # train the model
    trainer.train(script_args.resume_from_checkpoint)


if __name__ == "__main__":
    train_model()
