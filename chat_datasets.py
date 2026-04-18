"""
Datasets for supervised learning (SFT) that use chat-formatted data, which we
convert to tokens using a Renderer.
"""

#https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/chat_sl/chat_datasets.py

import logging
import hashlib
import re
import langid
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

FOREIGN_CHAR_PATTERN = re.compile(r"[\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\u0600-\u06FF]")

# Turns messages in the dataset into a list of strings
def _messages_to_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        content = str(message.get("content", ""))
        if content:
            parts.append(content)
    return "\n".join(parts)

# Function for casting the dict to a string
def _row_text(row: dict) -> str:
    messages = cast(list[dict], row.get("messages", []))
    return _messages_to_text(messages)

# Function for filtering foreign language examples (using langid)
def _is_foreign_example(row: dict) -> bool:
    classification = langid.classify(_row_text(row))
    return classification[0] != "en"

# Function for filtering math
def _is_math_example(row: dict) -> bool:
    return False

# Function for filtering Olmo/TableGPT examples
def _is_olmo_or_tablegpt(row: dict) -> bool:
    return False

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
    lowmath: bool = False
    noforeign: bool = False
    max_example_tokens: int | None = 8000
    no_olmo_tablegpt: bool = False
    math_keep_fraction: float = 0.40

    def _token_length(self, row: dict) -> int:
        text = _row_text(row)
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _keep_math_example(self, row: dict) -> bool:
        text = _row_text(row)
        digest = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        return score < self.math_keep_fraction

    # The filters are applied in the following order, and if any filter fails, the example is removed
    # Max length
    # No Foreign Language
    # No Olmo/TableGPT
    # Lowmath (with some examples kept based on math_keep_fraction)
    def _passes_filters(self, row: dict) -> bool:
        if self.max_example_tokens is not None and self._token_length(row) > self.max_example_tokens:
            return False

        if self.noforeign and _is_foreign_example(row):
            return False

       # if self.no_olmo_tablegpt and _is_olmo_or_tablegpt(row):
            #return False

       # if self.lowmath and _is_math_example(row) and not self._keep_math_example(row):
            #return False

        return True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset(
            "allenai/tulu-3-sft-olmo-2-mixture-0225",
            split="train",
            streaming=False  # explicit — forces full download
        )
        dataset = cast(datasets.Dataset, dataset)

        before_count = len(dataset)
        print(self.lowmath, self.noforeign, self.max_example_tokens, self.no_olmo_tablegpt)
        dataset = dataset.filter(self._passes_filters, desc="Applying TrainBuilder filters")
        after_count = len(dataset)
        logger.info(
            "Dataset filtered: %d -> %d (lowmath=%s, noforeign=%s, no_olmo_tablegpt=%s)",
            before_count,
            after_count,
            self.lowmath,
            self.noforeign,
            self.no_olmo_tablegpt,
        )

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
