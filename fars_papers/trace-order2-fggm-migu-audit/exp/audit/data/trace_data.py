# TRACE dataset loading and formatting utilities.
# Loads 8 TRACE tasks from JSON, formats into prompt/answer pairs for causal LM training.
# Follows the TRACE reference DataCollator logic (left-padding, prompt-masking in labels).

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

TASK_NAMES = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"
]

TASK_EPOCHS = {
    "C-STANCE": 5, "FOMC": 3, "MeetingBank": 7, "Py150": 5,
    "ScienceQA": 3, "NumGLUE-cm": 5, "NumGLUE-ds": 5, "20Minuten": 7,
}

DEFAULT_ORDER = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"
]

ORDER_2 = [
    "NumGLUE-cm", "NumGLUE-ds", "FOMC", "20Minuten",
    "C-STANCE", "Py150", "MeetingBank", "ScienceQA"
]


class TraceDataset(Dataset):
    def __init__(self, data_path: str, task_name: str, split: str = "train"):
        file_path = os.path.join(data_path, task_name, f"{split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.task_name = task_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "prompt": self.data[idx]["prompt"],
            "answer": self.data[idx]["answer"],
        }


@dataclass
class TraceDataCollator:
    tokenizer: Any
    max_prompt_len: int = 1024
    max_ans_len: int = 512
    pad_to_multiple_of: int = 8
    inference: bool = False

    def __call__(self, batch):
        sources = []
        gts = []
        tokenized_sources = []
        label_lens = []
        actual_max_len = 0
        limit_len = self.max_prompt_len + self.max_ans_len if not self.inference else self.max_prompt_len

        for instance in batch:
            instruction = instance["prompt"]
            label = instance["answer"]
            sources.append(instruction)
            gts.append(label)

            if not self.inference:
                tokenized_label = self._tokenize(
                    label, limit_len, add_eos_token=True
                )
                tokenize_source = self._tokenize(
                    instruction + label, limit_len, add_eos_token=True
                )
                label_lens.append(len(tokenized_label["input_ids"]))
                tokenized_sources.append(tokenize_source)
            else:
                tokenize_source = self._tokenize(
                    instruction, limit_len, add_eos_token=False
                )
                tokenized_sources.append(tokenize_source)

            if len(tokenize_source["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenize_source["input_ids"])

        actual_pad_len = (
            (actual_max_len + self.pad_to_multiple_of - 1)
            // self.pad_to_multiple_of
            * self.pad_to_multiple_of
        )

        for idx in range(len(tokenized_sources)):
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])
            tokenized_sources[idx]["input_ids"] = (
                [self.tokenizer.pad_token_id] * pad_len
                + tokenized_sources[idx]["input_ids"]
            )
            tokenized_sources[idx]["attention_mask"] = (
                [0] * pad_len + tokenized_sources[idx]["attention_mask"]
            )

            if not self.inference:
                label_len = label_lens[idx]
                label_mask_len = actual_pad_len - label_len
                tokenized_sources[idx]["labels"] = (
                    [-100] * label_mask_len
                    + tokenized_sources[idx]["labels"][-label_len:]
                )

        model_inputs = {
            "input_ids": torch.tensor(
                [s["input_ids"] for s in tokenized_sources]
            ),
            "attention_mask": torch.tensor(
                [s["attention_mask"] for s in tokenized_sources]
            ),
        }

        if not self.inference:
            model_inputs["labels"] = torch.tensor(
                [s["labels"] for s in tokenized_sources]
            )

        model_inputs["sources"] = sources
        if self.inference:
            model_inputs["gts"] = gts

        return model_inputs

    def _tokenize(self, sentence, cutoff_len, add_eos_token=True):
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_dataloaders(
    data_path: str,
    task_order: List[str],
    tokenizer: Any,
    split: str = "train",
    batch_size: int = 16,
    max_prompt_len: int = 1024,
    max_ans_len: int = 512,
    shuffle: bool = True,
    distributed: bool = False,
    local_rank: int = -1,
    inference: bool = None,
):
    dataloaders = {}
    if inference is None:
        inference = split in ("test",)
    collator = TraceDataCollator(
        tokenizer=tokenizer,
        max_prompt_len=max_prompt_len,
        max_ans_len=max_ans_len,
        inference=inference,
    )
    for task_name in task_order:
        dataset = TraceDataset(data_path, task_name, split)
        if distributed and not inference:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle and not inference:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloaders[task_name] = DataLoader(
            dataset,
            collate_fn=collator,
            sampler=sampler,
            batch_size=batch_size,
        )
    return dataloaders
