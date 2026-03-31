"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""
from tinker_cookbook import cli_utils, checkpoint_utils
from datetime import datetime
from tinker_cookbook.supervised.common import compute_mean_nll
from tqdm import tqdm
import logging
from pathlib import Path

import chz
from chat_datasets import TrainBuilder
from train import CLIConfig
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook import model_info
import tinker
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset


logger = logging.getLogger(__name__)

class SFTTrainer:
    def __init__(self, model, train_dataset, val_dataset, training_args, log_path, checkpoint_path=None):
        # Argument initializations for trainer class
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.log_path = str(Path(log_path).resolve())
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        service_client = tinker.ServiceClient(base_url=training_args.base_url)

        # Resume via TrainingClient API when a checkpoint path is provided.
        # Not sure if this works
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            self.training_client = service_client.create_training_client_from_state_with_optimizer(checkpoint_path['state_path'])
        else:
            self.training_client = service_client.create_lora_training_client(self.model, rank=training_args.lora_rank)
            logger.info("Starting a fresh LoRA training client")

    def train(self):
        num_batches = len(self.train_dataset)
        print(self.train_dataset)
        total_steps = num_batches * self.training_args.num_epochs

        # Intialize tokenizer and renderer
        logger.info("Using renderer: %s", self.training_args.renderer_name)

        # Initialize progress bar, aiming to implement similar to what they have in train.py from the cookbook
        progress_bar = tqdm(total=total_steps, desc="Training", unit="batch")

        logger.info(
            f"Training for {num_batches} batches x {self.training_args.num_epochs} epochs"
        )

        # For each epoch:
        for epoch in range(self.training_args.num_epochs):
            self.train_dataset.set_epoch(seed=epoch)
            logger.info(f"Starting epoch {epoch}")

            # For each batch in train_dataset:
            for batch_idx in range(num_batches):
                # Sets a metrics map to log metrics at each step
                metrics = {}
                # Uses get batch function to convert batch to proper format for training client
                step = epoch * num_batches + batch_idx

                # Save checkpoint every save_every steps (specified by config)
                if self.training_args.save_every > 0 and step > 0 and step % self.training_args.save_every == 0:
                    Path(self.log_path).mkdir(parents=True, exist_ok=True)
                    checkpoint_utils.save_checkpoint(
                        training_client=self.training_client,
                        name=f"{step:06d}",
                        log_path=self.log_path,
                        loop_state={"epoch": epoch, "batch": batch_idx},
                        kind="both",
                    )

                # TODO Check to make sure this calculation is right - I just have something random here to test
                # Code based on https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py
                lr_mult = max(0.0, 1.0 - (step / total_steps))  # Linear decay
                current_lr = self.training_args.learning_rate * lr_mult
                adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

                batch = self.train_dataset.get_batch(batch_idx)  
                fwd_bwd = self.training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim = self.training_client.optim_step(adam_params)
                
                metrics.update(optim.result().metrics)

                # Compute metrics (inspired by sl_loop as cited above)
                logprobs = [x["logprobs"] for x in fwd_bwd.result().loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in batch]
                nll = compute_mean_nll(logprobs, weights)

                # Calls update with tinker to update the model parameters based on the optim step
                metrics.update(
                    {
                        "num_sequences": len(batch),
                        "progress": step / total_steps,
                        "train_mean_nll": nll,
                        "learning_rate": current_lr,
                    }
                )

                # Add progress bar update to know what's happening in the loop
                progress_bar.set_postfix(metrics)
                progress_bar.update(1)


        # Saves the final checkpoint at the very end (does this need to be best? Or should I leave it as is)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        checkpoint_utils.save_checkpoint(
                training_client=self.training_client,
                name="final",
            log_path=self.log_path,
                loop_state={"batch": num_batches},
                kind="both",
            )
        logger.info(f"Training finished. Final checkpoint saved to: {self.log_path}")
            

def main(cli_config: CLIConfig):
    model = cli_config.model_name

    model_name = model.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{cli_config.dataset}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    repo_root = Path(__file__).resolve().parent
    if cli_config.log_path is not None:
        log_path_path = Path(cli_config.log_path)
    else:
        log_path_path = Path("runs") / run_name

    # Make relative log paths stable regardless of current working directory.
    if not log_path_path.is_absolute():
        log_path_path = repo_root / log_path_path
    log_path = str(log_path_path.resolve())

    # Ensure checkpoint/log directory exists before checkpoint_utils writes files.
    Path(log_path).mkdir(parents=True, exist_ok=True)

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)

    # Inspired by code in train.py from cookbook
    dataset_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=cli_config.train_on_what,
    )

    # Audrey's helper function to create the datasets
    dataset_builder = TrainBuilder(common_config=dataset_config)
    train_dataset, val_dataset = dataset_builder()

    if cli_config.max_examples is not None:
        logger.info(f"Limiting train dataset to {cli_config.max_examples} examples for debugging")
        capped_examples = min(cli_config.max_examples, len(train_dataset.hf_dataset))
        limited_hf_dataset = train_dataset.hf_dataset.select(range(capped_examples))
        train_dataset = SupervisedDatasetFromHFDataset(
            limited_hf_dataset,
            batch_size=train_dataset.batch_size,
            map_fn=train_dataset.map_fn,
        )

    checkpoint_path = checkpoint_utils.get_last_checkpoint(log_path)
    # Pass CLIConfig as training_args so fields like num_epochs are available in the trainer loop.
    trainer = SFTTrainer(model, train_dataset, val_dataset, cli_config, log_path, checkpoint_path=checkpoint_path)
    trainer.train()

    #args - batch size, learning rate, num epochs, 

    #base_url: str | None = None
    #log_path: str = "/tmp/tinker-examples/sl-loop"
    #model_name: str = "meta-llama/Llama-3.1-8B"
    #batch_size: int = 128
    #learning_rate: float = 1e-4
    #max_length: int = 32768
    #train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    #lora_rank: int = 32
    #save_every: int = 20  # 0 = disabled
    #ttl_seconds: int | None = 604800  # 7 days

    #dataset = SupervisedDatasetFromHFDataset(dataset_file, batch_size)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
