"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""
import asyncio
import inspect
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

wandb.init(project="test-project", config={"learning_rate": 1e-5, "batch_size": 248, "steps": 1500})


def _get_best_checkpoint_record(log_path: str) -> checkpoint_utils.CheckpointRecord | None:
    checkpoints = checkpoint_utils.load_checkpoints_file(log_path)
    best_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.state_path is not None
        and (checkpoint.get("best") is True or checkpoint.name.startswith("best-step-"))
    ]
    if not best_checkpoints:
        return None
    return best_checkpoints[-1]

class SFTTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args,log_path, load_checkpoint_path: str | None = None, resume_best_val_nll: float | None = None, resume_best_checkpoint_name: str | None = None):
        # Argument initializations for trainer class
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.log_path = log_path
        self.logged_weighted_example = False
        self.best_val_nll = (
            resume_best_val_nll if resume_best_val_nll is not None else float("inf")
        )
        self.best_checkpoint_name: str | None = resume_best_checkpoint_name

        service_client = tinker.ServiceClient(base_url=training_args.base_url)
        if load_checkpoint_path is not None:
            logger.info("Resuming training from checkpoint %s", load_checkpoint_path)
            self.training_client = service_client.create_training_client_from_state_with_optimizer(load_checkpoint_path)
        else:
            self.training_client = service_client.create_lora_training_client(self.model)

    async def _await_value(self, value):
        return await value

    def _resolve_result(self, value):
        while True:
            if inspect.isawaitable(value):
                value = asyncio.run(self._await_value(value))
                continue

            result_method = getattr(value, "result", None)
            if callable(result_method):
                resolved = result_method()
                if resolved is value:
                    return value
                value = resolved
                continue

            return value

    def _compute_validation_nll(self) -> float:
        total_weighted_logprobs = 0.0
        total_weights = 0.0

        for batch_idx in range(len(self.val_dataset)):
            batch = self.val_dataset.get_batch(batch_idx)
            if not batch:
                continue

            result = self._resolve_result(self.training_client.forward(batch, loss_fn="cross_entropy"))

            logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
            weights = [datum.loss_fn_inputs["weights"] for datum in batch]

            for logprobs_item, weights_item in zip(logprobs, weights, strict=True):
                logprobs_torch = logprobs_item.to_torch()
                weights_torch = weights_item.to_torch()
                total_weighted_logprobs += logprobs_torch.dot(weights_torch).item()
                total_weights += weights_torch.sum().item()

        if total_weights == 0:
            logger.warning("No valid weights found for validation NLL computation")
            return float("nan")

        return -total_weighted_logprobs / total_weights


    def _maybe_save_best_checkpoint(self, *, step: int, epoch: int, batch_idx: int) -> None:
        val_nll = self._compute_validation_nll()

        # Validate before logging or saving
        if math.isnan(val_nll):
            logger.info("Skipping best checkpoint save because validation NLL is NaN")
            return

        wandb.log({"val/nll": val_nll, "step": step})
        logger.info("Validation NLL at step %d: %.6f", step, val_nll)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
            self.best_checkpoint_name = f"best-step-{step:06d}"
            checkpoint_utils.save_checkpoint(
                training_client=self.training_client,
                name=self.best_checkpoint_name,
                log_path=self.log_path,
                loop_state={
                    "epoch": epoch,
                    "batch": batch_idx,
                    "step": step,
                    "best": True,
                    "val_mean_nll": val_nll,
                },
                kind="both",
            )
            logger.info(
                "Saved new best checkpoint %s with validation NLL %.6f",
                self.best_checkpoint_name,
                val_nll,
            )

    def train(self):
        num_batches = len(self.train_dataset)
        max_steps = self.training_args.max_steps

        # max_steps is interpreted as a per-epoch cap.
        if max_steps is not None:
            steps_per_epoch = max(1, min(num_batches, max_steps))
        else:
            steps_per_epoch = num_batches
        total_steps = max(1, self.training_args.num_epochs * steps_per_epoch)

        # Initialize progress bar, aiming to implement similar to what they have in train.py from the cookbook
        progress_bar = tqdm(total=total_steps, desc="Training", unit="batch")

        # Debugging statements for epochs
        logger.info(
            f"Training for {num_batches} batches x {self.training_args.num_epochs} epochs"
            + (f" (capped at {max_steps} steps per epoch)" if max_steps is not None else "")
        )

        last_completed_step: int | None = None
        last_completed_epoch = 0
        last_completed_batch = 0

        # For each epoch:
        for epoch in range(self.training_args.num_epochs):
            self.train_dataset.set_epoch(seed=epoch)
            logger.info(f"Starting epoch {epoch}")

            # For each batch in train_dataset:
            for batch_idx in range(num_batches):
                # Sets a metrics map to log metrics at each step
                metrics = {}
                step = epoch * steps_per_epoch + batch_idx
                batch = self.train_dataset.get_batch(batch_idx)

                # DEBUG - check to see if the statement is being weighted right
                if not self.logged_weighted_example and batch:
                    logger.info("Weighted example preview:\n%s", colorize_example(batch[0], self.tokenizer))
                    logger.info("Weights: %s", batch[0].loss_fn_inputs["weights"].tolist())
                    self.logged_weighted_example = True

                # Terminate the epoch loop if we've reached max steps for this epoch.
                if max_steps is not None and batch_idx >= max_steps:
                    logger.info(f"Reached max_steps={max_steps} for epoch={epoch}. Moving to next epoch.")
                    break

                # Code based on https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py
                progress = step / max(total_steps - 1, 1)
                lr = self.training_args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
                adam_params = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

                # Calls the pre cooked forward and backward pass with optim step
                fwd_bwd_results = self._resolve_result(
                    self.training_client.forward_backward(batch, loss_fn="cross_entropy")
                )

                optim_results = self._resolve_result(self.training_client.optim_step(adam_params))
                metrics.update(optim_results.metrics)

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

                # Log metrics for wandb, DEBUG
                wandb.log({"train/nll": nll, "learning_rate": lr, "step": step})

                # Add progress bar update to know what's happening in the loop
                progress_bar.set_postfix(metrics)
                progress_bar.update(1)

                last_completed_step = step
                last_completed_epoch = epoch
                last_completed_batch = batch_idx

                # Save checkpoints after the step so the saved state matches the updated weights.
                if self.training_args.save_every > 0 and step > 0 and step % self.training_args.save_every == 0:
                    checkpoint_utils.save_checkpoint(training_client=self.training_client, name=f"{step:06d}", log_path=self.log_path, loop_state={"epoch": epoch, "batch": batch_idx, "step": step}, kind="both")

        if last_completed_step is not None:
            self._maybe_save_best_checkpoint(
                step=last_completed_step,
                epoch=last_completed_epoch,
                batch_idx=last_completed_batch,
            )

        # Saves the final checkpoint at the very end (does this need to be best? Or should I leave it as is) TODO
        checkpoint_utils.save_checkpoint(training_client=self.training_client, name="final", log_path=self.log_path, loop_state={"batch": num_batches, "best_val_nll": self.best_val_nll, "best_checkpoint": self.best_checkpoint_name}, kind="both")
        logger.info("Training finished. Final checkpoint saved to: %s. Best checkpoint: %s", self.log_path, self.best_checkpoint_name)


def main(cli_config: CLIConfig):

        #hf_dataset: datasets.Dataset,
        #batch_size: int,
        #map_fn: Callable[[dict], tinker.Datum] | None = None,
        #flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,

    # Get logs stuff
    # build full config
    model_name = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{cli_config.dataset}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = str(Path("runs") / run_name)

    resume_checkpoint_record = _get_best_checkpoint_record(log_path)
    load_checkpoint_path = cli_config.load_checkpoint_path
    resume_best_val_nll: float | None = None
    resume_best_checkpoint_name: str | None = None

    if load_checkpoint_path is None and resume_checkpoint_record is not None:
        load_checkpoint_path = resume_checkpoint_record.state_path
        resume_best_checkpoint_name = resume_checkpoint_record.name
        best_val_nll = resume_checkpoint_record.get("val_mean_nll")
        if best_val_nll is not None:
            resume_best_val_nll = float(best_val_nll)

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
    # It wasn't working from your code so I think I'm just getting confused on how to use it
    # TODO: Tell me how this works maybe? I use it below
    print(cli_config.max_length)
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
    trainer = SFTTrainer(cli_config.model_name, tokenizer, train_dataset, val_dataset, cli_config, log_path, load_checkpoint_path=load_checkpoint_path, resume_best_val_nll=resume_best_val_nll, resume_best_checkpoint_name=resume_best_checkpoint_name)
    trainer.train()

    #args - batch size, learning rate, num epochs, 

    #dataset = SupervisedDatasetFromHFDataset(dataset_file, batch_size)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)