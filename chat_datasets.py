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
            logger.info(f"  {role}: {_shorten_text(content)}")


@chz.chz
class TrainBuilder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
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
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            val_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

