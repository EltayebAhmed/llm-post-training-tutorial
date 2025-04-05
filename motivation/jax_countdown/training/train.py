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

import itertools
import os
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import functools
import transformers

from flax import linen as nn
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils, train_state
import optax

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import TOKENS, REVERSE_TOKENS, get_data_loader, load_dataset_from_file, decode_tensor
from config import Config


def rsqrt_schedule(
    init_value: float,
    shift: int = 0,
):
    """Applies a reverse square-root schedule.

    The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

    Args:
      init_value: Base learning rate (before applying the rsqrt schedule).
      shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
        schedule makes it less steep in the beginning (close to 0).

    Returns:
      A schedule that applies the reverse square root.
    """

    def schedule(count):
        return init_value * (count + shift) ** -0.5 * shift**0.5

    return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
    """Creates a rsqrt schedule with linear warmup."""
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            ),
            rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
        ],
        boundaries=[warmup_steps],
    )


def compute_weighted_cross_entropy(
    logits: jnp.ndarray, targets: jnp.ndarray, pad_token: int,
    weights: Optional[jnp.ndarray] = None
):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
      logits: [batch, length, num_classes] float array.
      targets: categorical targets [batch, length] int array.
      pad_token: int, the padding token id.
    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    vocab_size = logits.shape[-1]

    # Shift targets by one for auto-regressive loss.
    targets = targets[:, 1:]
    logits = logits[:, :-1, :]

    # The mask is used to avoid training on padding tokens.
    mask = targets != pad_token

    mask = mask.astype(jnp.float32)[..., None]

    targets = common_utils.onehot(
        targets,
        vocab_size,
    )

    loss = -jnp.sum(targets * nn.log_softmax(logits) * mask, axis=-1)
    
    if weights is not None:
        loss = loss * weights[:, None]

    return loss.sum(), mask


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------

def train_step(
    state,
    inputs,
    dropout_rng: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    pad_token: int = -1,
):
    """Perform a single training step."""

    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        mask = inputs != pad_token
        position_ids = mask.cumsum(axis=-1, dtype=jnp.int32) - 1
        logits = state.apply_fn(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=mask,
             params=params, dropout_rng=dropout_rng, train=True
        ).logits

        loss, mask = compute_weighted_cross_entropy(
            logits=logits, targets=inputs, pad_token=pad_token,
            weights=weights
        )
        mean_loss = loss / mask.sum()
        return mean_loss, (logits, mask)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (mean_loss, (logits, mask)), grads = grad_fn(state.params)

    # Average gradients across GPUs
    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)

    return new_state, mean_loss, logits, mask


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=common_utils.shard_prng_key(self.dropout_rng)
        )


def train_and_evaluate(config: Config, workdir: str):
    """Runs a training and evaluation loop.

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
        train_ds, config.batch_size, config.max_target_length, config.max_prompt_length, shuffle=True
    )

    # Initialize Jax RNG states.
    rng = jax.random.PRNGKey(config.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Make sure different hosts have different dropout keys.
    dropout_rng = jax.random.split(dropout_rng, jax.local_device_count())

    print("Initializing model, optimizer, and step functions.")
    # Build Model and Optimizer
    # ---------------------------------------------------------------------------
    model_config = transformers.GPT2Config(
        vocab_size=len(TOKENS),
        n_positions=config.max_target_length,
        n_embd=config.emb_dim,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_inner=config.mlp_dim,
    )

    model = transformers.FlaxGPT2LMHeadModel(model_config, seed=config.seed)

    param_count = sum(p.size for p in jax.tree_leaves(model.params))
    print("Model has %d parameters.", param_count)

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

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

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

    infinite_iterator = itertools.cycle(train_data_loader)
    for step, batch_input_ids in enumerate(infinite_iterator):
        is_last_step = step == config.num_train_steps - 1

        # Shard data to devices and do a training step.

        batch_input_ids, _, _ = common_utils.shard(batch_input_ids)   
        state, train_loss, logits, mask = jit_train_step(state, batch_input_ids, dropout_rng)

        train_losses.append(train_loss.mean())
        # Quick indication that training is happening.

        if step < 5:
            print("Finished training step %d." % step)

        if step % config.log_every_steps == 0 or is_last_step:
            print("Step %d: Training Loss %.4f" %( step, train_losses[-1]))
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

    cfg = tyro.cli(Config)
    train_and_evaluate(cfg, cfg.save_dir)
