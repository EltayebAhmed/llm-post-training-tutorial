# Author: Eltayeb Ahmed
# Largely Adapted from https://github.com/google/flax/blob/main/examples/lm1b/train.py
# Original Acknowledgement:
# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language Modeling example.

This script trains a Transformer on a LM1B dataset.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import functools
import itertools
import os
import sys

from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import transformers

from config import ConfigRL
from preprocessing import (
    TOKENS,
    get_data_loader,
    load_dataset_from_file,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward import compute_reward
from sampling import sample
from train import TrainState, create_learning_rate_schedule, train_step
from utils import load_model, unshard


def rl_training_loop(config: ConfigRL, workdir: str):
    """Runs a RL training loop.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
            contains checkpoint training will be resumed from the latest checkpoint.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir)

    # Load Dataset
    # ---------------------------------------------------------------------------
    print("Initializing dataset.")
    train_ds = load_dataset_from_file(
        config.train_file,
    )

    train_data_loader = get_data_loader(
        train_ds,
        config.batch_size,
        config.max_target_length,
        config.max_prompt_length,
        shuffle=True,
    )

    # Initialize Jax RNG states.
    rng = jax.random.PRNGKey(config.seed)
    rng, dropout_rng, generation_rng = jax.random.split(rng, 3)

    # Make sure different hosts have different dropout keys.
    dropout_rng = jax.random.split(dropout_rng, jax.local_device_count())

    print("Initializing model, optimizer, and step functions.")
    # Build Model and Optimizer
    # ---------------------------------------------------------------------------
    # model_config = HashableGPT2Config.from_pretrained(config.base_model_path)
    # model = transformers.FlaxGPT2LMHeadModel(model_config, seed=config.seed)
    model = load_model(config.base_model_path)
    learning_rate_fn = create_learning_rate_schedule(
        learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
    )

    optimizer = optax.adamw(
        learning_rate_fn,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=config.weight_decay,
    )

    # checkpoint_path = os.path.abspath(config.base_model_path)
    # restored_params = checkpoints.restore_checkpoint(
    #     checkpoint_path,
    #     target=None,
    #     step=None,
    # )
    state = TrainState.create(
        apply_fn=model.__call__,
        # params=restored_params["params"],
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    param_count = sum(p.size for p in jax.tree_leaves(state.params))
    print("Model has %d parameters.", param_count)

    state = jax_utils.replicate(state)

    # compile multidevice versions of train/eval/predict step fn.
    jit_train_step = jax.pmap(
        functools.partial(
            train_step,
            pad_token=TOKENS["PAD"],
        ),
        donate_argnums=0,
        axis_name="batch",
    )

    # Main Train Loop
    # ---------------------------------------------------------------------------

    print("Starting training loop.")

    train_losses = []
    train_rewards = []

    infinite_iterator = itertools.cycle(train_data_loader)
    for step, batch in enumerate(infinite_iterator):
        is_last_step = step == config.num_train_steps - 1

        # Shard data to devices and do a training step.
        _, prompts, targets = batch

        generation_rng = jax.random.fold_in(generation_rng, step)

        parallel_generation_rng = jax.random.split(
            generation_rng, jax.local_device_count()
        )

        # replicate each prompt n_rollouts times
        prompts = jnp.repeat(prompts, config.num_rollouts, axis=0)
        targets = jnp.repeat(targets, config.num_rollouts, axis=0)

        generations = sample(
            model.config,
            transformers.FlaxGPT2LMHeadModel,
            state.params,
            common_utils.shard(prompts),
            TOKENS["PAD"],
            TOKENS["EOS"],
            parallel_generation_rng,
            config.generation_length,
            config.temperature,
        )
        generations = unshard(generations)

        rewards = compute_reward(
            generations,
            targets,
        )

        # log reward before processing it
        train_rewards.append(rewards.mean())

        # Normalize rewards. Recent research GRPO
        # (https://arxiv.org/pdf/2402.03300) shows that normalizing
        # rewards is improves training.
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        # rewards = rewards.astype(jnp.float32) / config.reward_scale

        # Compute Average of rollouts as baseline.
        rewards = rewards.reshape(config.num_rollouts, -1)

        rewards = rewards - rewards.mean(axis=0, keepdims=True)
        rewards = rewards.reshape(-1)

        generations = common_utils.shard(generations)
        rewards = common_utils.shard(rewards)

        state, train_loss, logits, mask = jit_train_step(
            state, generations, dropout_rng, weights=rewards
        )

        train_losses.append(train_loss.mean())

        # Quick indication that training is happening.
        if step < 5:
            print("Finished training step %d." % step)

        if step % config.log_every_steps == 0 or is_last_step or step < 5:
            print(
                "Step %d: Training Loss %.4f Reward %.4f "
                % (step, train_losses[-1], train_rewards[-1])
            )
            train_losses = []

        # Save a checkpoint on one host after every checkpoint_freq steps.
        save_checkpoint = step % config.checkpoint_every_steps == 0 or is_last_step
        if config.save_checkpoints and save_checkpoint:
            print("Saving checkpoint step %d.", step)
            to_checkpoint = jax_utils.unreplicate(state)
            checkpoints.save_checkpoint_multiprocess(workdir, to_checkpoint, step)

            checkpoint_dir = os.path.join(workdir, f"checkpoint_{step}")
            model.save_pretrained(checkpoint_dir, params=to_checkpoint.params)

        if is_last_step:
            return train_losses

    return train_losses


if __name__ == "__main__":
    import tyro

    print("Running training script.")

    cfg = tyro.cli(ConfigRL)
    rl_training_loop(cfg, cfg.save_dir)
