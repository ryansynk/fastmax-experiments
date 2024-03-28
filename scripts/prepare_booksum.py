import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from lightning_utilities.core.imports import RequirementCache
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
        destination_path: Path = Path("data/booksum"),
        checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
        data_file_url: str = "https://huggingface.co/datasets/kmfoda/booksum/resolve/main/",
        ignore_index: int = -1,
        max_seq_length: Optional[int] = None,
        mask_inputs: bool = False,
) -> None:
    """Prepare the Booksum dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    print("Loading data file...")
    download_if_missing(destination_path, data_file_url)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)
    data = read_data(destination_path)
    train_set, test_set = data["train"], data["test"]
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def download_if_missing(dest_dir: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""

    train_file_path = dest_dir / "train.csv"
    dev_file_path = dest_dir / "dev.csv"
    test_file_path = dest_dir / "test.csv"
    if train_file_path.exists() and dev_file_path.exists() and test_file_path.exists():
        return

    requests_available = RequirementCache("requests")
    if not requests_available:
        raise ModuleNotFoundError(str(requests_available))
    import requests
    for split in ["train", "dev", "test"]:
        file_path = dest_dir / f"{split}.csv"
        with open(file_path, "w", encoding="utf-8") as f:
            url = f"{file_url}{split}.csv?download=true"
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()
            f.write(response.text)


def read_data(dest_dir: Path) -> dict:
    """Reads the raw csv data file and returns a dictionary."""
    data = {}
    for split in ["train", "dev", "test"]:
        file_path = dest_dir / f"{split}.csv"
        d = pd.read_csv(file_path, usecols=["chapter", "summary_text"])
        d.rename(columns={"chapter": "input", "summary_text": "output"}, inplace=True)
        data[split] = d.to_dict(orient="records")

    data["train"] = data["train"] + data["dev"]
    del data["dev"]
    return data


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    example["instruction"] = "Write a summary for the following text."
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
