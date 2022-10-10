from huggingface_hub import HfFolder
from rm_runner import EC2RemoteRunner
import argparse

parser = argparse.ArgumentParser(description="Launches a model training job with Habana Gaudi on AWS.")
parser.add_argument("--input_tokenized_dataset_name", required=True)
parser.add_argument("--input_tokenizer_name", required=True)
parser.add_argument("--gaudi_config_id", required=True)
parser.add_argument("--output_model_name", required=True)
args = parser.parse_args()


hyperparameters = {
    "model_config_id": "roberta-large",
    "dataset_id": args.input_tokenized_dataset_name,
    "tokenizer_id": args.input_tokenizer_name,
    "gaudi_config_id": args.gaudi_config_id,
    "repository_id": args.output_model_name,
    "hf_hub_token": HfFolder.get_token(),  # need to be logged in with `huggingface-cli login`
    "max_steps": 100_000,
    "per_device_train_batch_size": 32,
    "learning_rate": 5e-5,
}
hyperparameters_string = " ".join(f"--{key} {value}" for key, value in hyperparameters.items())

runner = EC2RemoteRunner(
  instance_type="dl1.24xlarge",
  profile="hf-sm",  # adjust to your profile
  region="us-east-1",
  container="huggingface/optimum-habana:4.21.1-pt1.11.0-synapse1.5.0"
  )

runner.launch(
    command=f"python3 gaudi_spawn.py --use_mpi --world_size=8 run_mlm.py {hyperparameters_string}",
    source_dir="scripts",
)
