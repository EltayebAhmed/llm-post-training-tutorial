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
from transformers.agents import prompts

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
    def __init__(self, n_operands: int = 4, model_name: str = "model"):
        self.operand_log = [[] for i in range(n_operands)]
        self.rewards = []
        self.model_name = model_name

    def add(self, problem: list[int], reward: Optional[int] = None):

        try:
            eqn = extract_and_validate_equation(problem)
        except AssertionError:
            # Invalid equation, skip
            return
        operands = re.split(r"[*+-]", eqn)
        operands = [int(i) for i in operands]
        for i, operand in enumerate(operands):
            self.operand_log[i].append( operand)

        if reward is not None:
            self.rewards.append(reward)

    def plot_histograms_per_operand(self, n_bins: int = 20):
        """Plots histograms for each operand."""
        fig, axs = plt.subplots(len(self.operand_log), 1, figsize=(10, 10))
        fig.tight_layout(pad=3.0)
        for i, operand in enumerate(self.operand_log):
            axs[i].hist(operand, bins=n_bins, density=True)
            axs[i].set_xlabel(f"Operand {i + 1}")
            axs[i].set_ylabel("Density")
            mean = sum(operand) / len(operand)
            axs[i].set_title(f"{self.model_name} Histogram of Operand {i + 1}, {mean=:.2f}")

    def plot_reward_histogram(self, n_bins: int = 50):
        plt.hist(self.rewards, bins=n_bins, density=True)
        plt.xlabel("Reward")
        plt.ylabel("Density")
        mean = sum(self.rewards) / len(self.rewards)
        plt.title(f"{self.model_name} Reward Histogram, {mean=:.2f}")



DEFUALT_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data_test.csv")
DEFAULT_MODEL = os.path.abspath(DEFUALT_DATA)

def run_eval(
    model_path: str, data_path: str= DEFUALT_DATA, batch_size: int = 128, num_datapoints: int = 1024,
    model_name: str = "model",
    verbose: bool = False,
):

    dataset = load_dataset_from_file(data_path)

    dataset.data = dataset.data[:num_datapoints]

    dataloader = get_data_loader(
        dataset, batch_size=batch_size, seq_len=31, prompt_seq_len=15, shuffle=False
    )

    config = HashableGPT2Config.from_pretrained(
        model_path
    )
    model = transformers.FlaxGPT2LMHeadModel(config)

    prng = jax.random.PRNGKey(0)

    model_path = os.path.abspath(model_path)
    restored_params = checkpoints.restore_checkpoint(
        model_path,
        target=None,
        step=None,
    )
    model.params = restored_params["params"]

    report = OperandHistogramCounter(model_name=model_name)

    sum_ = 0
    count = 0
    for batch in dataloader:

        seq, prompt, res = batch

        prng = jax.random.fold_in(prng, 0)

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
        )
        seq_len = result.shape[-1]
        result = result.reshape(-1, seq_len)

        rewards = compute_reward(result, res).ravel()
        sum_ += rewards.sum().item()
        count += rewards.size

        if verbose:
            prompts = decode_tensor(prompt)
            prompts = [" ".join(p) for p in prompts]

            generations = decode_tensor(result)
            generations = [" ".join(p) for p in generations]

            for prompt, generation, reward in zip(prompts, generations, rewards.tolist()):
                print(f"Prompt: {prompt}\nGeneration: {generation}\nReward: {reward}")
                print()

        for row, reward in zip(result.tolist(), rewards.tolist()):
            report.add(row, reward)

    print(f"Averge reward:{sum_/count=} computed over {count} samples.")

    return report


if __name__ == "__main__":
    report = tyro.cli(run_eval)
    report.plot_histograms_per_operand()
    plt.savefig(f"{report.model_name}_operand_histogram.png")
    plt.figure()
    report.plot_reward_histogram()
    plt.xlabel("Reward")
    plt.savefig(f"{report.model_name}_reward_histogram.png")