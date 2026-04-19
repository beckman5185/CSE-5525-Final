"""
This module implements the PREFTrainer class for training your model using preference optimization.
"""

import asyncio
import logging
from pathlib import Path
from typing import cast

import chz
import tinker
import torch
import torch.nn.functional as F
from datetime import datetime

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset, ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir
from tinker_cookbook import model_info
from tinker_cookbook.weights import download, build_hf_model, publish_to_hf_hub, build_lora_adapter

from tinker_cookbook.preference.dpo_datasets import (
    DPODatasetBuilderFromComparisons,
)

from tinker_cookbook import checkpoint_utils, cli_utils, renderers

from chat_datasets import PrefBuilder

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub.hf_api import repo_exists
from huggingface_hub import create_repo
from peft import PeftModel


from tinker_cookbook.weights._merge import merge_adapter_weights

logger = logging.getLogger(__name__)


#also referenced Katie's train_sft implementation
#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/preference/dpo/train.py
#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/preference/train_dpo.py
#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tutorials/407_rlhf_pipeline.py 

@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "pref"  # or path like tinker_cookbook.preference.preference_datasets:HHHBuilder
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    dpo_beta: float = 0.1
    max_length: int | None = 8192 #did not appear to speed up for 4096
    batch_size: int = 128 #256
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 4 #did not appear to speed up for 2

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None

    # DPO-specific parameters
    reference_model_name: str | None = None #= "meta-llama/Llama-3.2-1B"

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None



def get_dataset_builder(
    dataset: str,
    model_name: str,
    renderer_name: str,
    max_length: int | None,
    batch_size: int,
) -> ChatDatasetBuilder:
    """Get the appropriate dataset builder for DPO training."""
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    logger.info(f"Renderer name: {renderer_name}")

    if dataset == "pref":
        return DPODatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=PrefBuilder()
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")



class PREFTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args, log_path, sft_path):

        #basic training args
        self.model=model
        self.tokenizer=tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.log_path = log_path
        self.logged_weighted_example = False
        self.load_checkpoint_path = training_args.load_checkpoint_path
        
        #if no checkpoint path given
        if training_args.load_checkpoint_path is None:
            #load checkpoint from SFT
            sft_ckpt = checkpoint_utils.get_last_checkpoint(str(Path(sft_path)))
            self.load_checkpoint_path = sft_ckpt.state_path 
            logger.info(f"SFT checkpoint loaded from {self.load_checkpoint_path}")

        #get renderer name
        #if none given, get recommended
        self.renderer_name = training_args.renderer_name or model_info.get_recommended_renderer_name(model)


    def train(self):
        #check that you have the right LORA rank
        logger.info(f"LORA rank: {self.training_args.lora_rank}")

        #set up DPO config
        config = train_dpo.Config(
        log_path=self.log_path,
        model_name=self.model,
        renderer_name=self.renderer_name,
        dataset_builder=get_dataset_builder(
            self.training_args.dataset,
            self.model,
            self.renderer_name,
            self.training_args.max_length,
            self.training_args.batch_size,
        ),
        load_checkpoint_path=self.load_checkpoint_path,
        evaluator_builders=[],
        learning_rate=self.training_args.learning_rate,
        lr_schedule=self.training_args.lr_schedule,
        num_epochs=self.training_args.num_epochs,
        dpo_beta=self.training_args.dpo_beta,
        lora_rank=self.training_args.lora_rank,
        base_url=self.training_args.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=self.training_args.wandb_name,
        reference_model_name=self.training_args.reference_model_name,
        max_steps=self.training_args.max_steps,
        loss_type="ipo",
        )


        #call training loop for DPO
        train_dpo.main(config)





def cli_main(cli_config: CLIConfig):
    """Main CLI function that builds the full config and calls the training function."""
    # Build full config
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.dataset}-{model_name}-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = str(Path("runs") / run_name)
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

        
    # Ensure checkpoint/log directory exists before checkpoint_utils writes files.
    Path(log_path).mkdir(parents=True, exist_ok=True)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)


    # Log the information about the run for debugging
    logger.info(
        "Run config: "
        f"model={cli_config.model_name}, dataset={cli_config.dataset}, "
        f"batch_size={cli_config.batch_size}, max_length={cli_config.max_length}, "
        f"num_epochs={cli_config.num_epochs}, max_steps={cli_config.max_steps}, "
        f"log_path={log_path}, load_checkpoint_path={cli_config.load_checkpoint_path}"
    )


    #get tokenizer
    tokenizer = get_tokenizer(cli_config.model_name) #AutoTokenizer.from_pretrained(cli_config.model_name)

    #not sure if this part is right. Datasets are different in this one
    #create datasets using builder
    dataset_builder = get_dataset_builder(cli_config.dataset, cli_config.model_name, 
                                          renderer_name, cli_config.max_length, cli_config.batch_size)
    train_dataset, val_dataset = dataset_builder()

    #hardcoding in the best one
    sft_log_path = "runs/sft-1-whole-set"

    #initialize trainer and train
    trainer = PREFTrainer(cli_config.model_name, tokenizer, train_dataset, val_dataset, cli_config, log_path, sft_log_path)
    trainer.train()


    #logging info about complete run
    logger.info(f"Model checkpoint saved to {log_path}")
    logger.info("IPO training completed successfully")



    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
