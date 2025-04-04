from typing import Callable
import torch
import torch.utils.data
import sys
import os
import jax.numpy as jnp
from typing import Callable
import numpy as np

# Gaurantees that the module can be imported from the parent directory.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

TOKENS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "-": 11,
    "*": 12,
    "=": 13,
    ":": 14,
    ",": 15,
    "EOS": 16,
    "BOS": 17,
    "PAD": 18,
}
REVERSE_TOKENS = {v: k for k, v in TOKENS.items()}


def tokenize(entry: str, add_eos_token: bool = True) -> jnp.ndarray:
    """Converts an equation to a sequence of tokens.

    Args:
        entry (str): The equation and result in the format "equation=result".

    Returns:
        jnp.ndarray: The sequence of tokens.
    """
    tokens = [TOKENS["BOS"]]
    for char in entry:
        if char in " \n":
            continue
        tokens.append(TOKENS[char])

    if add_eos_token:
        tokens.append(TOKENS["EOS"])

    return jnp.array(tokens)


# Example dataset class
class CountdownDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[str]):
        """Torch dataset for the example file.

        Args:
            data (list[tuple[str, str]]): Each entry in the list is
                an entry in the dataset. The tuple contains the equation
                and the result.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_example = self.data[idx]

        # full_example = tokenize(self.data[idx])
        split = full_example.split(":")
        prompt = ":".join(split[:2]) + ":"

        answer = jnp.array([int(split[0])])

        return tokenize(full_example), tokenize(prompt, add_eos_token=False), answer


def collate_single_tensor(
    batch: list[jnp.ndarray], seq_len: int, padding_side: str = "right"
) -> jnp.ndarray:
    """Collator function for the DataLoader.

    Args:
        batch (list[jnp.ndarray]): The batch to collate.

    Returns:
        jnp.ndarray: The collated batch.
    """
    max_len = max([len(x) for x in batch])
    assert max_len <= seq_len, f"All sequences must be shorter"
    f"than seq_len, sample with length found {max_len}, but seq_len "
    f"is {seq_len}."

    collated_batch = jnp.ones((len(batch), seq_len), dtype=jnp.int32) * TOKENS["PAD"]
    for i, x in enumerate(batch):
        if padding_side == "right":
            collated_batch = collated_batch.at[i, : len(x)].set(x)
        elif padding_side == "left":
            collated_batch = collated_batch.at[i, -len(x) :].set(x)

    return collated_batch


def get_collator_fn(
    full_seq_len: int, prompt_seq_len
) -> Callable[[list[jnp.ndarray]], tuple[jnp.ndarray, ...]]:
    """Returns a collator function for the DataLoader.

    This a Jax version of the collator function for the DataLoader. The collator
    function is used to collate the batch of data into a single tensor.
    For optimal performance with Jax, each batch should have the same length.
    We pad all batches to the same length with the pad_value."""

    def collator_fn(batch: list[jnp.ndarray]) -> tuple[jnp.ndarray, ...]:
        """Collator function for the DataLoader.

        Args:
            batch (list[jnp.ndarray]): The batch to collate.

        Returns:
            jnp.ndarray: The collated batch.
        """
        input_ids = collate_single_tensor(
            [x[0] for x in batch], full_seq_len, padding_side="right"
        )
        prompts = collate_single_tensor(
            [x[1] for x in batch], prompt_seq_len, padding_side="left"
        )

        answers = jnp.concatenate([x[2] for x in batch], axis=0)

        return input_ids, prompts, answers

    return collator_fn


def detokenize(tokens: list[int]):
    """Converts a sequence of tokens to a string.

    Args:
        tokens (list[int]): The sequence of tokens.

    Returns:
        str: The string representation of the tokens.
    """
    tokens_to_add_spaces_to = ["+", "-", "*", "=", "EOS", "PAD"]
    string = "".join([REVERSE_TOKENS[token] for token in tokens])

    for token in tokens_to_add_spaces_to:
        string = string.replace(token, f" {token} ")
    return string


def decode_tensor(token_ids: jnp.ndarray | list | int):
    """Decode token ids to strings."""
    if isinstance(token_ids, jnp.ndarray):
        return decode_tensor(token_ids.tolist())
    if isinstance(token_ids, list):
        return [decode_tensor(token_id) for token_id in token_ids]
    elif isinstance(token_ids, int):
        return REVERSE_TOKENS[token_ids]
    else:
        raise ValueError(
            "token_ids must be a list or a jnp.ndarray, not {}".format(type(token_ids))
        )


def load_dataset_from_file(data_path: str) -> CountdownDataset:
    """Loads the dataset from a CSV file.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        CountdownDataset: The dataset.
    """
    # csv = np.genfromtxt(data_path, delimiter=',', dtype=str).tolist()
    with open(data_path, "r") as f:
        lines = f.readlines()
    return CountdownDataset(lines)


def get_data_loader(
    dataset: CountdownDataset,
    batch_size: int,
    seq_len: int,
    prompt_seq_len: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Returns a DataLoader for the dataset.

    Args:
        dataset (CountdownDataset): The dataset.
        batch_size (int): The batch size.
        seq_len (int): The sequence length.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: The DataLoader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=get_collator_fn(seq_len, prompt_seq_len),
    )


# Example usage
if __name__ == "__main__":
    # Sample data
    data_path = (
        "/mount/llm-post-training-tutorial/motivation/jax_countdown/data/data_train.csv"
    )
    # Load the CSV file using numpy
    with open(data_path, "r") as f:
        dataset = f.readlines()
    # Create the dataset
    dataset = CountdownDataset(dataset)
    # Create DataLoader
    # print(dataset[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=get_collator_fn(20, 15),
        num_workers=2,
    )

    # Iterate through DataLoader
    for batch in dataloader:
        # print((batch))
        batch = [x.tolist() for x in batch]
        print(batch)
        for equation, prompt, res in zip(*batch):
            print(detokenize(equation))
            print(detokenize(prompt))
            print(res)
            print()
            input()
        quit()
