import os
import re

import sys

from flax.training import checkpoints
import jax
import jax.numpy as jnp
from torch import eq
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import (
    REVERSE_TOKENS,
    TOKENS,
    decode_tensor,
    get_data_loader,
    load_dataset_from_file,
)
from sampling import sample




def compute_reward(
    generations: jnp.ndarray,
    targets: jnp.ndarray,
    invalid=-40_000,
    clip: int = 100,
) -> jnp.ndarray:
    """Compute the rewards for the generations.
    
    Assumes operands are <10 and that there are 4 operands.
    Args:
        generations (jnp.ndarray): The generations to compute the rewards for.
        rewards (jnp.ndarray): The rewards to compute the rewards for.

    Returns:
        jnp.ndarray: The computed rewards.
    """
    tokens = generations

    tokens = tokens.tolist()
    tokens = [[REVERSE_TOKENS.get(i, "") for i in entry] for entry in tokens]

    tokens = ["".join(entry) for entry in tokens]
    # Remove the EOS and PAD tokens.
    tokens = [re.sub("EOS.*", "", x) for x in tokens]

    # Remove the question, leaving only the solution
    equations = [x.split(":") for x in tokens]
    results = []

    LARGE_VALUE = 10 ** 6
    for equation in equations:
        try:
            assert len(equation) == 3, "The solution is not in the right format."
            equation = equation[-1]

            assert re.match(r"^\d([*+-]\d){3}$", equation), "Equation is not valid."

            value = eval(equation)
            assert isinstance(value, int), "The result is not a number."
            # Avoid overflow and underflow. 
            value = min(value, LARGE_VALUE)
            value = max(value, -LARGE_VALUE)

        except Exception as e:
            value = invalid
        results.append(value)

    results = jnp.array(results)
    # Compute the rewards as the difference between the targets and the results.
    rewards = -jnp.abs(targets - results).clip(max=clip)
    rewards = jnp.where(rewards == invalid, -clip - 1, rewards)
    return rewards


if __name__ == "__main__":
    path = (
        "/mount/llm-post-training-tutorial/motivation/jax_countdown/data/data_train.csv"
    )
    dataset = load_dataset_from_file(path)
    dataloader = get_data_loader(dataset, batch_size=16, seq_len=31, prompt_seq_len=15)

    config = transformers.GPT2Config.from_pretrained("/mount/llm-post-training-tutorial/checkpoints/")
    model = transformers.FlaxGPT2LMHeadModel(config)
    state = checkpoints.restore_checkpoint(
        "/mount/llm-post-training-tutorial/checkpoint_post_fix/checkpoint_680",
        # "/mount/llm-post-training-tutorial/long_run/checkpoint_6780",
        target=None,
        step=None,
    )

    model.params = state["params"]

    prng = jax.random.PRNGKey(0)
    for batch in dataloader:

        seq, prompt, res = batch

        print(f"{prompt.shape=}")
        result = sample(
            model,
            prompt[:, :],
            TOKENS["PAD"],
            prng,
            9,
            temperature=0.1,
        )
        print(*enumerate(decode_tensor(result)), sep="\n")
        print(list(enumerate(compute_reward(result, res).tolist())))
        breakpoint()

        # for line in seq:
        #     line = line.tolist()
        #     line = [REVERSE_TOKENS.get(i, "") for i in line]
        #     print(" ".join(line))
        #     print(list(enumerate(line)))
        # index = int(input("Index to change: "))
        # value = int(input("Value to change to: "))
        # seq = seq.at[0, index].set(value)

        # print("Modified sequence:")
        # for line in seq:
        #     line = line.tolist()
        #     line = [REVERSE_TOKENS.get(i, "") for i in line]
        #     print(" ".join(line))
        #     print(list(enumerate(line)))
        # print(compute_reward(seq, res))
