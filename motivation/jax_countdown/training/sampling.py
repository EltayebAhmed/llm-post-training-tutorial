import functools
import os
import sys

from flax import jax_utils
from flax.training import checkpoints, common_utils
import jax
import jax.numpy as jnp
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import REVERSE_TOKENS, TOKENS, decode_tensor
from utils import HashableGPT2Config

@functools.partial(jax.pmap, static_broadcasted_argnums=(0, 1, 4, 5, 7, 8))
def sample(
    config,
    model_class,
    params,
    input_,
    padding_token,
    eos_token,
    rng,
    n_tokens,
    temperature=1.0,
):
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

    # Add batch dim that has been removed by pmap
    outputs = [input_]

    model = model_class(config)
    model.params = params

    batch_size = input_.shape[0]
    seq_len = input_.shape[1]
    max_seq_len = seq_len
    assert max_seq_len < 30, (
        "Initialisation kv_cache crashes"
        "on seq_len > 30, unknown why. Maybe config issue or a bug in library."
    )
    kv_cache = model.init_cache(batch_size, max_length=max_seq_len)

    mask = input_ != padding_token
    position_ids = (mask).cumsum(axis=1) - 1

    preload = model(
        input_, position_ids=position_ids, attention_mask=mask, past_key_values=kv_cache
    )

    logits = preload.logits
    past_key_values = preload.past_key_values

    rolling_tokens = jax.random.categorical(rng, logits[:, -1] / temperature, axis=-1)[
        :, None
    ]
    rolling_position_ids = position_ids[:, [-1]] + 1

    outputs.append(rolling_tokens)

    is_finished = jnp.zeros_like(rolling_tokens)

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
        sampled_tokens = jax.random.categorical(
            rng, logits[:, i - 1] / temperature, axis=-1
        )[:, None]

        # sampled_tokens = sampled_tokens * (1 - is_finished) + padding_token * is_finished
        sampled_tokens = jnp.where(is_finished, padding_token, sampled_tokens)

        is_finished = jnp.logical_or(is_finished, sampled_tokens == eos_token)
        outputs.append(sampled_tokens)
        rolling_tokens = sampled_tokens

        rolling_position_ids = rolling_position_ids + 1

    output = jnp.concatenate(outputs, axis=-1)
    return output


if __name__ == "__main__":
    # Example usage
    rng = jax.random.PRNGKey(0)

    config = HashableGPT2Config.from_pretrained(
        "/mount/llm-post-training-tutorial/checkpoints/"
    )

    state = checkpoints.restore_checkpoint(
        "/mount/llm-post-training-tutorial/checkpoint_post_fix/checkpoint_680",
        # "/mount/llm-post-training-tutorial/long_run/checkpoint_6780",
        target=None,
        step=None,
    )

    model = transformers.FlaxGPT2LMHeadModel(config)
    model.params = state["params"]

    # "/mount/llm-post-training-tutorial/checkpoints/"
    batch_size = 1024
    input_ = jnp.zeros((batch_size, 1), dtype=jnp.int32)  # Example input tensor

    input_ = input_ + TOKENS["BOS"]  # Set all elements to the token for 'BOS'

    input_ = common_utils.shard(input_)
    rng = jax.random.split(rng, jax.local_device_count())

    # with jax.disable_jit():

    import time

    state = jax_utils.replicate(state)

    print("starting jit")
    # results_classic = sample_classic(
    #     model,
    #     input_,
    #     TOKENS["PAD"],
    #     rng,
    #     25,
    #     0.01
    # )
    results = sample(
        config,
        transformers.FlaxGPT2LMHeadModel,
        state["params"],
        input_,
        TOKENS["PAD"],
        TOKENS["EOS"],
        rng,
        25,
        0.6,
    )
    # print("sample_classic")
    # print(*decode_tensor(results_classic[0, :5]), sep="\n")
    print("sample")
    print(*decode_tensor(results[0, :5]), sep="\n")
    breakpoint()
    print("ending jit")
    print(results.ravel()[0])
    stime = time.time()
    results = sample(
        config,
        transformers.FlaxGPT2LMHeadModel,
        state["params"],
        input_,
        TOKENS["PAD"],
        rng,
        25,
        0.01,
    )
    print(results.ravel()[0])
    print("computed vals in ", time.time() - stime)

    results = results.reshape(batch_size, -1)
    print(results)

    for i in range(4):
        print(
            [REVERSE_TOKENS[token] for token in results[i].tolist()]
        )  # Convert token IDs to strings
    breakpoint()
    # print(tokenized_entry)  # Output: [17, 3, 10, 5, 13, 8, 16]
