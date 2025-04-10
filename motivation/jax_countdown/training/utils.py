"""Some utitilites that will be useful for training and evaluation."""

import os

from flax import jax_utils
from flax.training import checkpoints, train_state
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import transformers


class HashableGPT2Config(transformers.GPT2Config):
    """A hashable version of the GPT2Config class.

    This can be passed as a static_arg to a jax jitted function."""

    def __hash__(self):
        return id(self)


def compute_parameter_mean(model):
    """Computes the mean of model parameters.

    We use this to compare whether or not two sets of model
    paramers are identical. This very crude hashing mechanism is
    not perfect, but should help catch at least some bugs.
    """
    param_means = [x.mean() for x in jax.tree_util.tree_leaves(model.params)]
    return sum(param_means).item() / len(param_means)


class TrainState(train_state.TrainState):
    """A train state which has dropout rng.

    It contains model parameters, optimizer state and dropout rng."""

    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=common_utils.shard_prng_key(self.dropout_rng)
        )


def load_model(checkpoint_path: str):
    """Load a model from a Flax checkpoint.

    We assume the checkpoint contains a full TrainState
    that includes the model parameters and optimizer state.
    We will initialise a mirroring train state then load the
    parameters and optimizer state from the checkpoint before
    discarding the optimizer state.

    """

    checkpoint_path = os.path.abspath(checkpoint_path)

    # Instantiate the model
    config = HashableGPT2Config.from_pretrained(
        checkpoint_path,
    )
    model = transformers.FlaxGPT2LMHeadModel(config) # type: ignore

    # Instantiate Optimizer to and full training state.
    optimizer = optax.adamw(
        learning_rate=0,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=0,
    )

    train_state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=jax.random.PRNGKey(0),
    )

    train_state = vars(train_state)
    del train_state["apply_fn"], train_state["step"], train_state["tx"]

    # We compute a crude hash of the randomly initialised
    # model parameters.
    initialised_param_mean = compute_parameter_mean(model)

    # Restore params from checkpoint.
    restored = checkpoints.restore_checkpoint(
        os.path.abspath(checkpoint_path),
        target=train_state,
        step=None,
    )

    model.params = restored["params"]

    # Make sure the hash changed and that we did not
    # encounter a bug. Flax seems to change the checkpointing
    # API every other week. Sigh.
    loaded_param_mean = compute_parameter_mean(model)

    assert (
        abs(initialised_param_mean - loaded_param_mean) > 1e-4
    ), "Model params did not change after loading. Loading failes"
    return model


def unshard(x: jnp.ndarray):
    """Unshard a jax array.

    Undo the sharding of tensors that was 
    result of using jax.pmap.

    Args:
        x: A jax array.

    Returns:
        The unsharded array.
    """
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), x)
