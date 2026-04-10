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
import datetime

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset, ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

from tinker_cookbook.preference.dpo_datasets import (
    DPODatasetBuilderFromComparisons,
)

from tinker_cookbook import checkpoint_utils, cli_utils, renderers

from chat_datasets import PrefBuilder

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


#also referenced Katie's train_sft implementation
#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/preference/dpo/train.py
#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/preference/train_dpo.py

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
    max_length: int | None = 8192
    batch_size: int = 256
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 16

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 20
    infrequent_eval_every: int = 100
    inline_evals: str | None = None

    # DPO-specific parameters
    reference_model_name: str | None = None

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

    if dataset == "pref":
        return DPODatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=PrefBuilder()
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the DPO loss and associated training metrics.

    Implements the loss from *Direct Preference Optimization* (Rafailov et al., 2023):
    ``L = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))``."""
      
    #compute log ratios
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    #compute dpo loss
    #review formula - ????
    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

    #compute metrics
    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": loss.item(),
        "accuracy": accuracy,
        "margin": margin, 
        "chosen reward": chosen_rewards.mean().item(),
        "rejected reward": rejected_rewards.mean().item()
    }

    return loss, metrics


def do_update(
    epoch_idx: int,
    batch_idx: int, 
    n_batches: int, 
    total_steps: int,
    config: CLIConfig,
    training_client: tinker.TrainingClient, 
    reference_client: tinker.SamplingClient,
    evaluators: list[Evaluator], 
    infrequent_evaluators: list[Evaluator],
    dataset: SupervisedDatset, 
    ml_logger: ml_log.Logger, 
    log_path: str, 
    tokenizer: Tokenizer, 
    rolling_mgr: checkpoint_utils.RollingCheckpointManager | None = None
):
    """Perform a single DPO training update step.

    Handles checkpointing, evaluation, reference log-prob computation,
    the forward-backward pass with the custom DPO loss, the optimizer step,
    and metric logging for one batch."""

    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    with trace.trace_iteration(step=step) as window:
        #save checkpoint if needed
        #rolling manager - ?

        learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=config.lr_schedule, step=step, total_steps=total_steps
        )
        adam_params = tinker.AdamParams(
            learning_rate=learning_rate, 
            #store Adam params?
            beta1=None, 
            beta2=None, 
            eps=None
        )

        #do evaluation

        #Prepare batch
        #what is trace doing here?
        with trace.scope_span_sync("get_batch"):
            data = dataset.get_batch(batch_idx)

        #Split data into chosen and rejected pairs
        #does this work?
        chosen_data = [datum for i, datum in enumerate(data) if i%2==0]
        rejected_data = [datum for i, datum in enumerate(data) if i%2==1]

        #print example for first batch

        #what is trace doing here?
        with trace.scope_span_sync("get_ref_logprobs"):
            #get reference log probs
            #reconstruct full sequences for sampling client
            full_sequences = []
            for datum in data:
                #Reconstruct full sequence by appending last target token
                target_tokens = datum.loss_fn_inputs["target_tokens"].data
                if target_tokens:
                    #does my object have correct structure? surely
                    full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
                    full_sequences.append(full_sequence)
                else:
                    #if no target, use model input as-is
                    full_sequences.append(datum.model_input)

            #compute reference log probabilities in parallel
            async def compute_all_ref_logprobs():
                return await asyncio.gather(
                    *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
                )
            
            all_ref_logprobs = asyncio.run(compute_all_ref_logprobs())

            #extract relevant logprobs (skip first token, prompt)
            all_ref_logprob_seqs = [torch.tensor(logprobs[1:]) for logprobs in all_ref_logprobs]

            #split reference results into chosen and rejected
            chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
            rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

            #create DPO loss functions
            def dpo_loss_fn(
                data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
            ) -> tuple[torch.Tensor, dict[str, float]]:
                #split logprobs into chosen and rejected
                chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
                rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

                #extract logprobs
                chosen_logprobs = []
                chosen_ref_logprobs = []
                rejected_logprobs = []
                rejected_ref_logprobs = []

                for i in range(len(chosen_data)):
                    #compute weighted logprobs for chosen responses
                    chosen_logprob_seq = chosen_logprob_seqs[i]
                    chosen_ref_logprob_seq = chosen_ref_logprob_seqs[i]
                    chosen_weights = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
                    #is this the right dot product?
                    chosen_logprob = torch.dot(chosen_logprob_seq.float(), chosen_weights.float())
                    chosen_ref_logprob = torch.dot(
                        chosen_ref_logprob_seq.float(), chosen_weights.float()
                    )
                    chosen_logprobs.append(chosen_logprob)
                    chosen_ref_logprobs.append(chosen_ref_logprob)

                    #compute weighted logprobs for rejected responses
                    rejected_logprob_seq = rejected_logprob_seqs[i]
                    rejected_ref_logprob_seq = rejected_ref_logprob_seqs[i]
                    rejected_weights = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
                    rejected_logprob = torch.dot(rejected_logprob_seq.float(), rejected_weights.float())
                    rejected_ref_logprob = torch.dot(
                        rejected_ref_logprob_seq.float(), rejected_weights.float()
                    )
                    rejected_logprobs.append(rejected_logprob)
                    rejected_ref_logprobs.append(rejected_ref_logprob)

                #compute DPO loss
                return compute_dpo_loss(
                    chosen_logprobs = chosen_logprobs, 
                    rejected_logprobs=rejected_logprobs, 
                    chosen_ref_logrpobs=chosen_ref_logprobs,
                    rejected_ref_logprobs=rejected_ref_logprobs,
                    dpo_beta=config.dpo_beta
                )
            
            with trace.scope_span_sync("step"):
                #do forward-backward with custome DPO loss
                backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).reuslt()
                dpo_metrics = backward_result.metrics

                #optimizer step
                #where are adam_params from?
                training_client.optim_step(adam_params).result()

            
            #prepare metrics
            #where is metrics coming from?
            metrics.update(
                num_pairs=len(chosen_data), 
                num_tokens=sum(datum.model_input.length for datum in data), 
                learning_rate=learning_rate,
                progress=step/total_steps,
                **dpo_metrics
            )
        
        #additional logging



class PREFTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args, log_path):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.log_path = log_path
        self.logged_weighted_example = False

        service_client = tinker.ServiceClient(base_url=training_args.base_url)
        #added rank here - is this correct?
        #resume checkpoint stuff - see SFT example
        self.training_client = service_client.create_lora_training_client(self.model, rank=training_args.lora_rank)
        #do I need a reference client? I think so for DPO specifically
        self.reference_client = self.training_client.save_weights_and_get_sampling_client()

    def train(self):
        # This part is all from the tinker cookbook, preference, train_dpo.py
        # Implement the training loop here

        start_epoch = 0
        start_batch = 0

        #set up logger and tracing - ?

        #self.tokenizer, self.train_dataset, self.val_dataset

        #get number of batches
        n_batches = len(self.train_dataset)

        #get total number of steps
        #where is this used?
        total_steps = n_batches * self.training_args.num_epochs
        if self.training_args.max_steps is not None:
            total_steps = min(total_steps, self.training_args.max_steps)

        #evaluators?

        reached_max_steps = False
        for epoch_idx in range(start_epoch, self.training_args.num_epochs):
            #shuffle dataset
            self.train_dataset.set_epoch(seed=epoch_idx)

            for batch_idx in range(start_batch if epoch_idx==start_epoch else 0, n_batches):
                step = epoch_idx * n_batches + batch_idx
                if self.training_args.max_steps is not None and step > self.training_args.max_steps:
                    reached_max_steps = True
                    break

                #need to implement do_update
                do_update(
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    n_batches=n_batches,
                    total_steps=total_steps,
                    config=self.training_args,
                    training_client=self.training_client,
                    reference_client=self.reference_client,
                    #what are the evaluators?
                    evaluators=None,
                    infrequent_evaluators=None,
                    dataset=self.train_dataset,
                    #?
                    ml_logger=None,
                    log_path=self.training_args.log_path,
                    tokenizer=self.tokenizer,
                    #?
                    rolling_mgr=None
                )

                if reached_max_steps:
                    break

            #save final checkpoint

            #close logger
            

        pass


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
        log_path = f"/runs/{run_name}"
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
        
    # Ensure checkpoint/log directory exists before checkpoint_utils writes files.
    Path(log_path).mkdir(parents=True, exist_ok=True)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)


    #I think I was supposed to add logging in here somewhere?

    #get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cli_config.model_name)

    #not sure if this part is right. Datasets are different in this one
    #create datasets using builder
    dataset_builder = get_dataset_builder(cli_config.dataset, cli_config.model_name, 
                                          cli_config.renderer_name, cli_config.max_length, cli_config.batch_size)
    train_dataset, val_dataset = dataset_builder()

    #initialize trainer and train
    trainer = PREFTrainer(cli_config.model_name, tokenizer, train_dataset, val_dataset, cli_config, log_path)
    trainer.train()

    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)


#check back - definitely missing logging, checkpoints
#compare to recent SFT code
#do validation during training