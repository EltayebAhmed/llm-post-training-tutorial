import os
import re

import sys
from typing import Optional

from flax import jax_utils
from flax.training import checkpoints, common_utils, train_state
import jax
import jax.numpy as jnp
import optax
import transformers
import collections
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import (
    REVERSE_TOKENS,
    TOKENS,
    decode_tensor,
    get_data_loader,
    load_dataset_from_file,
)
from sampling import sample, HashableGPT2Config
from reward import compute_reward, extract_and_validate_equation
import tyro

class OperandHistogramCounter:
    def __init__(self, n_operands: int = 4):
        self.counts = [collections.defaultdict(int) for _ in range(n_operands)]
        self.rewards = []

    def add(self, problem: list[int], reward: Optional[int] = None):

        try:
            eqn = extract_and_validate_equation(problem)
        except AssertionError:
            # Invalid equation, skip
            return
        operands = re.split(r"[*+-]", eqn)
        operands = [int(i) for i in operands]
        for i, operand in enumerate(operands):
            self.counts[i][operand] += 1

        if reward is not None:
            self.rewards.append(reward)

    def print_report(self, norm=True):
        for i, counter in enumerate(self.counts):
            print(f"Operand {i}:")
            denominator = sum(counter.values()) if norm else 1

            for operand, count in sorted(counter.items()):
                print(f"  {operand}: {count / denominator :.4f}")
            print()

    def plot_reward_histogram(self):

        plt.hist(self.rewards, bins=50)
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        mean = sum(self.rewards) / len(self.rewards)
        plt.title(f"Reward Histogram, {mean=:.2f}")
        plt.show()

        plt.savefig("reward_histogram.png")



DEFUALT_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data_test.csv")
DEFAULT_MODEL = os.path.abspath(DEFUALT_DATA)

def run_eval(
    model_path: str, data_path: str= DEFUALT_DATA, batch_size: int = 128, num_datapoints: int = 1024
):

    dataset = load_dataset_from_file(data_path)

    dataset.data = dataset.data[:num_datapoints]

    dataloader = get_data_loader(
        dataset, batch_size=batch_size, seq_len=31, prompt_seq_len=15, shuffle=False
    )

    config = HashableGPT2Config.from_pretrained(
        # "/mount/llm-post-training-tutorial/checkpoints/"
        model_path
    )
    model = transformers.FlaxGPT2LMHeadModel(config)

    # model_path = "/mount/llm-post-training-tutorial/rl_bbb/checkpoint_4"
    # model_path = "/mount/llm-post-training-tutorial/checkpoint_post_fix/checkpoint_680"

 

    prng = jax.random.PRNGKey(0)

    model_path = os.path.abspath(model_path)
    restored_params = checkpoints.restore_checkpoint(
        model_path,
        target=None,
        step=None,
    )
    model.params = restored_params["params"]

    histogram = OperandHistogramCounter()
    sum_ = 0
    count = 0
    for batch in dataloader:

        seq, prompt, res = batch

        prng = jax.random.fold_in(prng, 0)

        print(f"{prompt.shape=}")
        result = sample(
            model.config,
            model.__class__,
            jax_utils.replicate(model.params),
            common_utils.shard(prompt),
            TOKENS["PAD"],
            TOKENS["EOS"],
            jax.random.split(prng, jax.local_device_count()),
            9,
            0.01,
        ).reshape(batch_size, -1)

        rewards = compute_reward(result, res).ravel()
        sum_ += rewards.sum().item()
        count += rewards.size

        for row, reward in zip(result.tolist(), rewards.tolist()):
            histogram.add(row, reward)


        print(f"{sum_=}, {count=}, {sum_/count=}")
    histogram.plot_reward_histogram()
    # histogram.print_report()

if __name__ == "__main__":
    tyro.cli(run_eval)