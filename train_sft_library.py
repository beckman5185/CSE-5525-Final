"""
This module implements supervised fine-tuning (SFT) via tinker_cookbook's built-in train loop. Most of this code is based on train.py
"""
import asyncio
from datetime import datetime
from pathlib import Path
import logging

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from train import CLIConfig
from chat_datasets import TrainBuilder

logger = logging.getLogger(__name__)


class SFTTrainer:
    def __init__(self, config: supervised_train.Config):
        self.config = config

    def train(self):
        # Runs the async training loop instead of doing it myself :/
        asyncio.run(supervised_train.main(self.config))


# Main function entry point
def main(cli_config: CLIConfig):
    # Loads in the model name
    model_name = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{cli_config.dataset}-{model_name}-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    )

    # Checks to see if we're resuming or not
    # Built with the help of Copilot in order to make sure I'm actually resuming from something
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = str(Path("runs") / run_name)

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # If user specifies renderer name use that, otherwise use get recommended renderer
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(cli_config.model_name)

    # Build config for chat dataset
    dataset_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=cli_config.train_on_what,
    )

    # Connect wandb for progress checking
    wandb_name = cli_config.wandb_name or run_name

    # Builds the training config with all of the different parameters specified
    # Checked with Copilot/inspired by train.py
    config = supervised_train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=TrainBuilder(common_config=dataset_config),
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        infrequent_eval_every=cli_config.infrequent_eval_every,
        renderer_name=renderer_name,
        max_steps=cli_config.max_steps * cli_config.num_epochs if cli_config.max_steps is not None else None,
    )

    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
