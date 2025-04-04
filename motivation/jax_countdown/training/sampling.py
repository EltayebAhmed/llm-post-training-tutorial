import re
import transformers
import os
import sys

from transformers.models.sew.modeling_sew import _SEQ_CLASS_CHECKPOINT
from vllm import inputs

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import TOKENS, REVERSE_TOKENS

import jax.numpy as jnp
import jax
import torch
from flax.training import checkpoints
import functools

# # @functools.partial(jax.jit, static_argnums=(0, 3))
# def sample(model, input_, rng, n_tokens):
#     def scan_fn(carry, ys):
#         results, rng, i = carry
#         logits = model(results).logits
#         rng = jax.random.fold_in(rng, 0)  # Fold in a dummy value since scan provides no index
#         sampled_tokens = jax.random.categorical(
#             rng, logits[:, i-1], axis=-1
#         )
#         results = results.at[:, i].set(sampled_tokens)
#         # results = jnp.concatenate([results, sampled_tokens], axis=-1)
#         return (results, rng, i + 1), None

#     original_length = input_.shape[1]
#     length = original_length + n_tokens
#     results = jnp.zeros((input_.shape[0], length), dtype=jnp.int32)
#     results = jax.lax.dynamic_update_slice(results, input_, (0, 0))

#     print(results.shape, 'before')
#     (results, rng, i), _ = jax.lax.scan(scan_fn, (input_, rng, original_length ), None, length=n_tokens)
#     print(results.shape, 'after')
#     return results


@functools.partial(jax.jit, static_argnums=(0, 2, 4))
def sample(model, input_, padding_token, rng, n_tokens, temperature=1.0):
    """Sample from the model using the given input as prompt.

    Uses kv_caching to speed up the sampling process.
    Args:
        model: The model to sample from.
        input_: The tokens to prompt to the model.
        padding_token: The token to use for padding.
        rng: The random key to use for sampling.
        n_tokens: The number of tokens to sample.
    Returns:
        The sequence of tokens sampled from the model. The
            prompt tokens are included in the output.
    """
    outputs = [input_]

    batch_size = input_.shape[0]
    seq_len = input_.shape[1]
    assert seq_len < 30, (
        "Initialisation kv_cache crashes"
        "on seq_len > 30, unknown why. Maybe config issue or a bug in library."
    )
    kv_cache = model.init_cache(batch_size, max_length=seq_len)

    mask = input_ != padding_token
    position_ids = (mask).cumsum(axis=1) - 1
    

    preload = model(input_, position_ids=position_ids,
    attention_mask=mask, past_key_values=kv_cache)

    logits = preload.logits
    past_key_values = preload.past_key_values

    rolling_tokens = jax.random.categorical(rng, logits[:, -1], axis=-1)[:, None]
    rolling_position_ids = position_ids[:, [-1]] + 1

    outputs.append(rolling_tokens)

    for i in range(n_tokens):
        results = model(
            input_ids=rolling_tokens,
            position_ids=rolling_position_ids,
            past_key_values=past_key_values,
        )

        past_key_values = results.past_key_values
        logits = results.logits

        rng = jax.random.fold_in(
            rng, 0
        )  # Fold in a dummy value since scan provides no index
        sampled_tokens = jax.random.categorical(rng, logits[:, i - 1] / temperature, axis=-1)[:, None]
        outputs.append(sampled_tokens)
        rolling_tokens = sampled_tokens

        rolling_position_ids = rolling_position_ids + 1

    output = jnp.concatenate(outputs, axis=-1)
    print(output.shape, "after")
    return output


if __name__ == "__main__":
    # Example usage
    config = transformers.GPT2Config.from_pretrained(
        "/mount/llm-post-training-tutorial/checkpoints/"
    )
    model = transformers.FlaxGPT2LMHeadModel(config)
    state = checkpoints.restore_checkpoint(
        # "/mount/llm-post-training-tutorial/checkpoint_post_fix/checkpoint_680",
        "/mount/llm-post-training-tutorial/long_run/checkpoint_6780",
        target=None,
        step=None,
    )

    model.params = state["params"]
    # "/mount/llm-post-training-tutorial/checkpoints/"
    input_ = torch.zeros((1024, 1), dtype=torch.int32)  # Example input tensor
    input_ = jnp.zeros((1024, 1), dtype=jnp.int32)  # Example input tensor

    input_ = input_ + TOKENS["BOS"]  # Set all elements to the token for 'BOS'
    results = input_

    # with jax.disable_jit():
    rng = jax.random.PRNGKey(0)

    import time

    print("starting jit")
    results = sample(model, input_, TOKENS["PAD"], rng, )
    print("ending jit")
    print(results.ravel()[0])
    stime = time.time()
    results = sample(model, input_, TOKENS["PAD"], rng, 25)
    print(results.ravel()[0])
    print("computed vals in ", time.time() - stime)

    print(results)
    for i in range(4):
        print(
            [REVERSE_TOKENS[token] for token in results[i].tolist()]
        )  # Convert token IDs to strings
    # print(tokenized_entry)  # Output: [17, 3, 10, 5, 13, 8, 16]
