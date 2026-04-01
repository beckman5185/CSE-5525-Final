"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""
# TODO: Check to make sure all imports are used
import math

from tinker_cookbook import cli_utils, checkpoint_utils
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from datetime import datetime
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook import model_info
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import chz
from chat_datasets import TrainBuilder
from train import CLIConfig
import tinker

logger = logging.getLogger(__name__)

import wandb

wandb.init(
    project="your-project-name",
    name="run-name",  # optional but helpful
    config={
        "learning_rate": 1e-5,
        "batch_size": 248,
        "steps": 1500,
    }
)

class SFTTrainer:
    def __init__(self, model, train_dataset, val_dataset, training_args, log_path, checkpoint_path=None):
        # Argument initializations for trainer class
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.log_path = log_path

        service_client = tinker.ServiceClient(base_url=training_args.base_url)
        self.training_client = service_client.create_lora_training_client(self.model)

        # Resume via TrainingClient API when a checkpoint path is provided.
        # Not sure if this works
        Path(log_path).mkdir(parents=True, exist_ok=True)

        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            self.training_client.load_state_with_optimizer(checkpoint_path).result()
        else:
            logger.info("Starting a fresh LoRA training client")

    def train(self):
        num_batches = len(self.train_dataset)
        planned_total_steps = self.training_args.num_epochs * num_batches
        max_steps = self.training_args.max_steps
        if max_steps is not None:
            total_steps = max(1, min(planned_total_steps, max_steps))
        else:
            total_steps = max(1, planned_total_steps)

        # Initialize progress bar, aiming to implement similar to what they have in train.py from the cookbook
        progress_bar = tqdm(total=total_steps, desc="Training", unit="batch")

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
                # Uses get batch function to convert batch to proper format for training client
                batch = self.train_dataset.get_batch(batch_idx)
                step = epoch * num_batches + batch_idx

                if max_steps is not None and step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}. Stopping training loop.")
                    stop_training = True
                    break

                # Save checkpoint every save_every steps (specified by config)
                if self.training_args.save_every > 0 and step > 0 and step % self.training_args.save_every == 0:
                    checkpoint_utils.save_checkpoint(
                        training_client=self.training_client,
                        name=f"{step:06d}",
                        log_path=self.log_path,
                        loop_state={"epoch": epoch, "batch": batch_idx},
                        kind="both",
                    )

                # TODO Check to make sure this calculation is right - I just have something random here to test
                # Code based on https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py
                progress = step / max(total_steps - 1, 1)
                lr = self.training_args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
                adam_params = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

                # Calls the pre cooked forward and backward pass with optim step
                fwd_bwd = self.training_client.forward_backward(batch, loss_fn="cross_entropy")
                fwd_bwd_results = fwd_bwd.result()  # Wait for forward and backward to complete before stepping the optimizer
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


        # Saves the final checkpoint at the very end (does this need to be best? Or should I leave it as is)
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

    #add data preprocessing - ?

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

    # Unify resume handling in one branch: if the run directory already has checkpoints and no explicit
    # checkpoint path was provided, use this run directory as the checkpoint source.
    checkpoint_path = cli_config.load_checkpoint_path
    if checkpoint_path and not checkpoint_path.startswith("tinker://") and Path(checkpoint_path).exists():
        resume_info = checkpoint_utils.get_last_checkpoint(checkpoint_path)
        if resume_info is not None:
            checkpoint_path = resume_info.state_path

    if not checkpoint_path and (Path(log_path) / "checkpoints.jsonl").exists():
        resume_info = checkpoint_utils.get_last_checkpoint(log_path)
        if resume_info is not None:
            checkpoint_path = resume_info.state_path

    logger.info(
        "Run config: "
        f"model={cli_config.model_name}, dataset={cli_config.dataset}, "
        f"batch_size={cli_config.batch_size}, max_length={cli_config.max_length}, "
        f"num_epochs={cli_config.num_epochs}, max_steps={cli_config.max_steps}, "
        f"save_every={cli_config.save_every}, log_path={log_path}"
    )
    if checkpoint_path:
        logger.info(f"Resolved resume checkpoint: {checkpoint_path}")
    else:
        logger.info("No checkpoint found for resume; training from scratch")

    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
    
    # Build the dataset builder config to pass to the TrainBuilder, which will create the train and val datasets (changed from Audrey's)
    # It wasn't working from your code so I think I'm just getting confused on how to use it
    # TODO: Tell me how this works maybe? I use it below
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