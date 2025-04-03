import numpy as np
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import tyro


### Data generation utilities.
@dataclasses.dataclass
class Distribution:
    """Distribution over a consecutive integer range."""

    min: int
    max: int
    probabilities: list[float]

    def __post_init__(self):
        """Check that the probabilities are valid."""
        assert len(self.probabilities) == self.max - self.min + 1
        assert np.isclose(sum(self.probabilities), 1)


@dataclasses.dataclass
class DataPoint:
    """A data point representing an equation."""

    equation: list[str]
    result: int


def generate_data(
    n_operands: int, n_samples: int, distributions: list[Distribution], seed: int = 42
) -> list[DataPoint]:
    """Generate random data according to the given distributions."""
    assert len(distributions) == n_operands

    operators = ["*", "+", "-"]
    available_ops = len(operators)
    data = []

    rng = np.random.RandomState(seed)

    for _ in range(n_samples):
        sample_equation = []
        for i, distribution in enumerate(distributions):
            operand = rng.choice(
                range(distribution.min, distribution.max + 1),
                p=distribution.probabilities,
            )
            sample_equation.append(str(operand))
            if i < n_operands - 1:
                sample_equation.append(operators[rng.randint(0, available_ops)])

        result = eval(" ".join(sample_equation))
        data.append(DataPoint(sample_equation, result))
    return data


def exponential_distribution(min: int, max: int, alpha: float) -> Distribution:
    """Multinomail distribution over a consecutive integer range.

    Probabilities are exponentially decreasing/increasing as we
        progress across integers.

    Args:
        min: Minimum integer value.
        max: Maximum integer value.
        alpha: Exponential decay/increase rate. Negative values will
            result in increasing probabilities, positive values in
            decreasing probabilities.
    """
    probabilities = [np.exp(-alpha * x) for x in range(min, max + 1)]
    probabilities = [p / sum(probabilities) for p in probabilities]
    return Distribution(min, max, probabilities)


def linear_growing_distribution(min: int, max: int, ratio: float) -> Distribution:
    """Multinomail distribution over a consecutive integer range.

    Probabilities are linearly increasing as we progress across integers.

    Args:
        min: Minimum integer value.
        max: Maximum integer value.
        ratio: Ratio between p_min and p_max.
    """
    probabilities = np.linspace(1, ratio, max - min + 1)
    probabilities = probabilities / probabilities.sum()
    probabilities = probabilities.tolist()

    return Distribution(min, max, probabilities)


def generate_dataset_with_skewed_operands(
    n_samples: int,
    n_operands: int,
    min_operand: int,
    max_operand: int,
    alpha: float = 4.0,
    seed: int = 42,
) -> list[DataPoint]:
    """Generate a dataset with skewed operands."""

    assert n_operands > 1
    EPS = 1e-6
    assert alpha > -EPS, "Alpha must be non-negative."
    assert min_operand < max_operand, "Min operand must be less than max operand."

    distributions = []
    # log_linspace = np.logspace(np.log10(alpha), np.log10(1 / alpha), n_operands)
    for alpha in np.logspace(np.log10(alpha), np.log10(1 / alpha), n_operands):
        distributions.append(
            linear_growing_distribution(min_operand, max_operand, alpha)
        )

    return generate_data(n_operands, n_samples, distributions, seed)


def write_dataset_to_disk(data: list[DataPoint], output_filepath: str):
    """Write the dataset to disk."""
    assert output_filepath.endswith(".csv"), "Output path must end with .csv"
    with open(output_filepath, "w") as f:
        for data_point in data:
            operands = [i for i in data_point.equation if i.isdigit()]

            problem = f"{data_point.result}: {','.join(operands)} "
            solution = " ".join(data_point.equation)
            f.write(f"{problem} : {solution}\n")


### Data visualization utilities.


def visualize_operand_histogram(data: list[DataPoint], n_bins: int = 100):
    """Visualize the per position histogram of operands in the dataset."""
    assert data, "Empty dataset."
    n_operands = len(data[0].equation) // 2 + 1

    for data_point in data:
        assert len(data_point.equation) == 2 * n_operands - 1

    fig, axs = plt.subplots(n_operands, 1, figsize=(10, 30))
    for i in range(n_operands):
        operands = [int(data_point.equation[2 * i]) for data_point in data]
        axs[i].hist(operands, bins=n_bins)
        axs[i].set_title(f"Operand {i + 1}")

