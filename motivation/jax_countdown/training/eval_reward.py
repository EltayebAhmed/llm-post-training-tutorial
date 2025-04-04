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
from reward import compute_reward



if __name__ == "__main__":
    path = (
        "/mount/llm-post-training-tutorial/motivation/jax_countdown/data/data_test.csv"
    )
    dataset = load_dataset_from_file(path)
    batch_size = 128
    num_datapoints = 1024

    dataset.data = dataset.data[:num_datapoints]

    dataloader = get_data_loader(dataset, batch_size=batch_size, seq_len=31, prompt_seq_len=15,
                                 shuffle=False)

    config = HashableGPT2Config.from_pretrained("/mount/llm-post-training-tutorial/checkpoints/")
    model = transformers.FlaxGPT2LMHeadModel(config)

    model_path = "/mount/llm-post-training-tutorial/rl_bbb/checkpoint_4"


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
        # print(*enumerate(decode_tensor(result)), sep="\n")
        # print(list(enumerate(compute_reward(result, res).tolist())))
        rewards = compute_reward(result, res)
        sum_ += rewards.sum().item()
        count += rewards.size

        print(f"{sum_=}, {count=}, {sum_/count=}")