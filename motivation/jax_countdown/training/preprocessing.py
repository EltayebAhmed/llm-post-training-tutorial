from typing import Callable
import torch
import torch.utils.data 
import sys
import os
import jax.numpy as jnp
from typing import Callable
import numpy as np

# Gaurantees that the module can be imported from the parent directory.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

TOKENS = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '+': 10,
    '-': 11,
    '*': 12,
    '=': 13,
    ":": 14,
    ",": 15,
    'EOS': 16,
    'BOS': 17,
    'PAD': 18,
}
REVERSE_TOKENS = {v: k for k, v in TOKENS.items()}

def represent_equation(equation: str, result: str) -> jnp.ndarray:
    """Converts an equation to a sequence of tokens.
    
    Args:
        equation (str): The equation to convert.
    
    Returns:
        jnp.ndarray: The sequence of tokens.
    """
    tokens = []
    for char in equation:
        if char == ' ':
            continue
        tokens.append(TOKENS[char])
    tokens.append(TOKENS['='])

    result = result.strip()
    for char in result:
        tokens.append(TOKENS[char])

    tokens.append(TOKENS['EOS'])

    return jnp.array(tokens)

# Example dataset class
class CountdownDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[tuple[str, str]]):
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
        return represent_equation(self.data[idx][0], self.data[idx][1])

def get_collator_fn(seq_len: int) -> Callable[[list[jnp.ndarray]], jnp.ndarray]:
    """Returns a collator function for the DataLoader.

    This a Jax version of the collator function for the DataLoader. The collator
    function is used to collate the batch of data into a single tensor.
    For optimal performance with Jax, each batch should have the same length.
    We pad all batches to the same length with the pad_value."""
    def collator_fn(batch: list[jnp.ndarray]) -> jnp.ndarray:
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

        collated_batch = jnp.ones((len(batch), seq_len)) * TOKENS['PAD']
        for i, x in enumerate(batch):
            collated_batch = collated_batch.at[i, :len(x)].set(x)
        return collated_batch
    return collator_fn

def detokenize(tokens:list[int]):
    """Converts a sequence of tokens to a string.
    
    Args:
        tokens (list[int]): The sequence of tokens.
    
    Returns:
        str: The string representation of the tokens.
    """
    tokens_to_add_spaces_to = ['+', '-', '*', '=', 'EOS', 'PAD']
    string = ''.join([REVERSE_TOKENS[token] for token in tokens])

    for token in tokens_to_add_spaces_to:
        string = string.replace(token, f' {token} ')
    return string

def load_dataset_from_file(data_path: str) -> CountdownDataset:
    """Loads the dataset from a CSV file.
    
    Args:
        data_path (str): The path to the CSV file.
    
    Returns:
        CountdownDataset: The dataset.
    """
    csv = np.genfromtxt(data_path, delimiter=',', dtype=str).tolist()
    return CountdownDataset(csv)

def get_data_loader(dataset: CountdownDataset,
                    batch_size: int,
                    seq_len: int,
                    shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Returns a DataLoader for the dataset.
    
    Args:
        dataset (CountdownDataset): The dataset.
        batch_size (int): The batch size.
        seq_len (int): The sequence length.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        
    Returns:
        torch.utils.data.DataLoader: The DataLoader.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=get_collator_fn(seq_len)) 
# Example usage
if __name__ == "__main__":
    # Sample data
    data_path = "/mount/AIMS-material/motivation/data.csv"
    # Load the CSV file using numpy
    dataset = np.genfromtxt(data_path, delimiter=',', dtype=str).tolist()
    # Create the dataset
    dataset = CountdownDataset(dataset)
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
                                             collate_fn=get_collator_fn(20),
                                             num_workers=2)

    # Iterate through DataLoader
    for batch in dataloader:
        print((batch))
        batch = batch.tolist()
        for equation in batch:
            print(detokenize(equation))
        quit()