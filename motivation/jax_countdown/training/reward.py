import os
import re

import sys

from flax import jax_utils
from flax.training import checkpoints, common_utils, train_state
import jax
import jax.numpy as jnp
import optax
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import (
    REVERSE_TOKENS,
    TOKENS,
    decode_tensor,
    get_data_loader,
    load_dataset_from_file,
)
from sampling import sample, HashableGPT2Config


def extract_and_validate_equation(equation: list[int]):
    processed = [REVERSE_TOKENS.get(i, "") for i in equation]

    processed = "".join(processed) 
    # Remove the EOS and PAD tokens.
    processed = re.sub("EOS.*", "", processed)

    # Remove the question, leaving only the solution
    processed = processed.split(":") 
    assert len(processed) == 3, "The solution is not in the right format."

    processed = processed[-1]
    assert re.match(r"^\d([*+-]\d){3}$", processed), "Equation is not valid."

    return processed

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

    if generations.ndim == 1:
        # Add batch dimension
        generations = generations[None, :]

    assert generations.ndim == 2, "Generations should be 2D."

    equations = generations.tolist()
    results = []

    LARGE_VALUE = 10 ** 6
    for equation in equations:
        try:
            equation = extract_and_validate_equation(equation)
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
    rewards = jnp.where(results == invalid, -clip - 1, rewards)
    return rewards


if __name__ == "__main__":
    path = (
        "/mount/llm-post-training-tutorial/motivation/jax_countdown/data/data_train.csv"
    )
    dataset = load_dataset_from_file(path)
    batch_size = 16
    dataloader = get_data_loader(dataset, batch_size=batch_size, seq_len=31, prompt_seq_len=15)

    config = HashableGPT2Config.from_pretrained("/mount/llm-post-training-tutorial/checkpoints/")
    model = transformers.FlaxGPT2LMHeadModel(config)
    model_path = "/mount/llm-post-training-tutorial/checkpoint_post_fix/checkpoint_680"
    
    # state = checkpoints.restore_checkpoint(
    #     model_path,
    #     target=None,
    #     step=None,
    # )
    # model.params = state["params"]

    optimizer = optax.adamw(
        learning_rate=0.0016,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=0.01,
    )

    prng = jax.random.PRNGKey(0)

    restored_params = checkpoints.restore_checkpoint(
        model_path,
        target=None,
        step=None,
    )
    state = train_state.TrainState.create(
        apply_fn=model.__call__,
        params=restored_params["params"],
        tx=optimizer,
    )

    model.params = state.params


    
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
            0.05,
        ).reshape(batch_size, -1)
        # print(*enumerate(decode_tensor(result)), sep="\n")
        # print(list(enumerate(compute_reward(result, res).tolist())))
        print(compute_reward(result, res))
        print(*decode_tensor(result), sep="\n")
        breakpoint()
