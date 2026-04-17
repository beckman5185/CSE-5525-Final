"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""

# Imports for the cookbook and other key utilities
import math

from tinker_cookbook import cli_utils, checkpoint_utils
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from datetime import datetime
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.display import colorize_example
from tinker_cookbook import model_info
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import chz
from chat_datasets import TrainBuilder
from train import CLIConfig
import tinker
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

import wandb

# Initializes wandb. Change this code to correspond to current run metrics
wandb.init(
    project="test-project",
    config={
        "learning_rate": 1e-5,
        "batch_size": 248,
        "steps": 1500,
    }
)

# Main class for training
class SFTTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args, log_path):
        # Argument initializations for trainer class
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.log_path = log_path
        self.logged_weighted_example = False

        # Creates the service and LoRA training clients using cookbook methods
        service_client = tinker.ServiceClient(base_url=training_args.base_url)
        self.training_client = service_client.create_lora_training_client(self.model)

    def train(self):
        # Gets number of batches and the epochs for the dataset
        num_batches = len(self.train_dataset)
        planned_total_steps = self.training_args.num_epochs * num_batches
        max_steps = self.training_args.max_steps
        
        # Adjust steps based on command line arguments, in case incorrect input is given
        if max_steps is not None:
            total_steps = max(1, min(planned_total_steps, max_steps))
        else:
            total_steps = max(1, planned_total_steps)

        # Initialize progress bar, aiming to implement similar to what they have in train.py from the cookbook
        progress_bar = tqdm(total=total_steps, desc="Training", unit="batch")

        # Debugging statements for epochs
        logger.info(
            f"Training for {num_batches} batches x {self.training_args.num_epochs} epochs"
            + (f" (capped at {max_steps} steps)" if max_steps is not None else "")
        )

        stop_training = False

        # For each epoch:
        for epoch in range(self.training_args.num_epochs):
            self.train_dataset.set_epoch(seed=epoch)
            logger.info(f"Starting epoch {epoch}")

            # For each batch in train_dataset:
            for batch_idx in range(num_batches):
                # Sets a metrics map to log metrics at each step
                metrics = {}
                step = epoch * num_batches + batch_idx
                batch = self.train_dataset.get_batch(batch_idx)

                # Gives weighted example preview for debug
                if not self.logged_weighted_example and batch:
                    logger.info("Weighted example preview:\n%s", colorize_example(batch[0], self.tokenizer))
                    logger.info("Weights: %s", batch[0].loss_fn_inputs["weights"].tolist())
                    self.logged_weighted_example = True

                # Terminate the loop if we've reached the max steps
                if max_steps is not None and step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}. Stopping training loop.")
                    stop_training = True
                    break

                # Save checkpoint every save_every steps (specified by config), inspired by cookbook
                if self.training_args.save_every > 0 and step > 0 and step % self.training_args.save_every == 0:
                    checkpoint_utils.save_checkpoint(
                        training_client=self.training_client,
                        name=f"{step:06d}",
                        log_path=self.log_path,
                        loop_state={"epoch": epoch, "batch": batch_idx},
                        kind="both",
                    )

                # Code based on https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py
                progress = step / max(total_steps - 1, 1)
                lr = self.training_args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
                adam_params = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

                # Calls the pre cooked forward and backward pass with optim step
                fwd_bwd = self.training_client.forward_backward(batch, loss_fn="cross_entropy")
                # Fix for potential async issues, may not be necessary
                fwd_bwd_results = fwd_bwd.result()  

                # Calls the optim step and updates metrics
                optim = self.training_client.optim_step(adam_params)
                metrics.update(optim.result().metrics)

                # Compute metrics (inspired by sl_loop as cited above)
                logprobs = [x["logprobs"] for x in fwd_bwd_results.loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in batch]
                nll = compute_mean_nll(logprobs, weights)

                # Calls update with tinker to update the model parameters based on the optim step
                metrics.update(
                    {
                        "num_sequences": len(batch),
                        "progress": step / total_steps,
                        "train_mean_nll": nll,
                        "learning_rate": lr,
                    }
                )

                # Log metrics for wandb to see how training is progressing and debug any model issues
                wandb.log({
                    "train/nll": nll,
                    "learning_rate": lr,
                    "step": step
                })

                # Add progress bar update to know what's happening in the loop
                progress_bar.set_postfix(metrics)
                progress_bar.update(1)

            if stop_training:
                break

        # Saves the final checkpoint at the very end
        checkpoint_utils.save_checkpoint(
                training_client=self.training_client,
                name="final",
            log_path=self.log_path,
                loop_state={"batch": num_batches},
                kind="both",
            )
        logger.info(f"Training finished. Final checkpoint saved to: {self.log_path}")
            

def main(cli_config: CLIConfig):

        #hf_dataset: datasets.Dataset,
        #batch_size: int,
        #map_fn: Callable[[dict], tinker.Datum] | None = None,
        #flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,

    #from sl_loop.py
    model = "meta-llama/Llama-3.2-1B"

    # Get logs stuff
    # build full config
    model_name = model.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{cli_config.dataset}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = str(Path("runs") / run_name)

    # Ensure checkpoint/log directory exists before checkpoint_utils writes files.
    Path(log_path).mkdir(parents=True, exist_ok=True)

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    
    # Log the information about the run for debugging
    logger.info(
        "Run config: "
        f"model={cli_config.model_name}, dataset={cli_config.dataset}, "
        f"batch_size={cli_config.batch_size}, max_length={cli_config.max_length}, "
        f"num_epochs={cli_config.num_epochs}, max_steps={cli_config.max_steps}, "
        f"save_every={cli_config.save_every}, log_path={log_path}"
    )

    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cli_config.model_name)
    
    # Build the dataset builder config to pass to the TrainBuilder, which will create the train and val datasets (changed from Audrey's)
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

    # Pass CLIConfig as training_args so fields like num_epochs are available in the trainer loop.
    trainer = SFTTrainer(model, tokenizer, train_dataset, val_dataset, cli_config, log_path)
    trainer.train()

    #args - batch size, learning rate, num epochs, 

    #dataset = SupervisedDatasetFromHFDataset(dataset_file, batch_size)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
