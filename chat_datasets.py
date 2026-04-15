"""
Datasets for supervised learning (SFT) that use chat-formatted data, which we
convert to tokens using a Renderer.
"""

#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/chat_sl/chat_datasets.py

import logging
from typing import cast

import chz
import datasets
import tinker

from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset



#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/preference/datasets.py

from tinker_cookbook import renderers
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
)



logger = logging.getLogger(__name__)


def _shorten_text(text: str, limit: int = 200) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def _log_conversation_preview(messages: list[dict], example_idx: int) -> None:
    logger.info(f"Conversation preview #{example_idx}")
    for message in messages:
        role = message.get("role", "unknown")
        content = str(message.get("content", ""))
        if role in {"user", "assistant"}:
            logger.info(f"  {role}: {content}")


@chz.chz
class TrainBuilder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset(
            "allenai/tulu-3-sft-olmo-2-mixture-0225",
            split="train",
            streaming=False  # explicit — forces full download
        )
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset.shuffle(seed=0)
        val_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        # Print a tiny preview so training logs show both user and assistant responses.
        preview_count = min(2, len(train_ds))
        for i in range(preview_count):
            row = train_ds[i]
            _log_conversation_preview(cast(list[dict], row["messages"]), i)

        # Use train_on_what from common_config if provided, otherwise default to LAST_ASSISTANT_MESSAGE
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        # take the last 1000 as test, the rest as train
        print(self.common_config.max_length)
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            val_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )




@chz.chz
class PrefBuilder(ComparisonDatasetBuilder):
    """Olmo 2 1B preference dataset comparison builder."""

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        #load in dataset
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", 
            split="train",
            streaming=False
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        #first 1024 examples as test
        test_dataset = dataset.take(1024)
        #rest as train
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        #get instruction, chosen response, and rejected response
        instruction = example["chosen"][0]["content"]
        chosen_response = example["chosen"][1]["content"]
        rejected_response = example["rejected"][1]["content"]

        #create conversation
        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        #make comparison object from conversation and the two responses
        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        #label comparisons (A is preferred)
        return LabeledComparison(comparison=comparison, label="A")