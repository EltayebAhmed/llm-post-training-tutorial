import dataclasses
import os
import sys

import matplotlib.pyplot as plt
import tyro

# Guarantees that the module can be imported from the parent directory.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_utils import (
    generate_dataset_with_skewed_operands,
    visualize_operand_histogram,
    write_dataset_to_disk,
)


@dataclasses.dataclass
class GenerationConfig:
    """Configuration for data generation.

    Will generate a skewed dataset with operands exponentially distributed
        with alpha values ranging from -alpha to alpha.
    """

    n_samples: int
    n_operands: int
    min_operand: int
    max_operand: int
    alpha: float
    seed: int
    output_filepath: str



if __name__ == "__main__":
    config = tyro.cli(GenerationConfig)
    dataset = generate_dataset_with_skewed_operands(
        n_samples=config.n_samples,
        n_operands=config.n_operands,
        min_operand=config.min_operand,
        max_operand=config.max_operand,
        alpha=config.alpha,
        seed=config.seed,
    )
    write_dataset_to_disk(dataset, config.output_filepath)
    n_bins = (config.max_operand - config.min_operand + 1) * 2 

    visualize_operand_histogram(dataset, n_bins=n_bins)
    plt.savefig(config.output_filepath.replace(".csv", ".png"))

