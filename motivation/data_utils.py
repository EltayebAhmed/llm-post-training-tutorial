import numpy as np
import dataclasses
import tyro

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
    n_operands: int, n_samples: int, distributions: list[Distribution],
    seed: int = 42
) -> list[list[int]]:
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

def generate_dataset_with_skewed_operands(n_samples: int, 
    n_operands: int, 
    min_operand: int,
    max_operand: int,
    alpha: float = 0.14,
    seed: int = 42):
    """Generate a dataset with skewed operands."""

    assert n_operands > 1
    EPS = 1e-6
    assert alpha > -EPS, "Alpha must be non-negative."
    assert min_operand < max_operand, "Min operand must be less than max operand."

    distributions = []
    for alpha in np.linspace(-alpha, alpha, n_operands):
        distributions.append(exponential_distribution(min_operand, max_operand, alpha))
    
    return generate_data(n_operands, n_samples, distributions, seed)

def main():
    import matplotlib.pyplot as plt
    import os
    print(os.getcwd())
    plt.plot(exponential_distribution(0, 10, -0.14).probabilities)
    plt.savefig("exponential_distribution.png")

def generate_data
if __name__ == "__main__":
    main()
