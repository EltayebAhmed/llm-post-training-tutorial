# Author: Eltayeb Ahmed
# Largely Adapted from https://github.com/google/flax/blob/main/examples/lm1b/
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

"""Default Hyperparameter configuration."""

import dataclasses
import os
from typing import Optional

TRAIN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "data_train.csv"
)


@dataclasses.dataclass(kw_only=True, frozen=True)
class Config:
    """Hyperparameter configuration for language model training."""

    train_file: str = TRAIN_FILE
    validation_file: Optional[str] = None
    vocab_size: int = 30_000

    num_train_steps: int = 500_000
    learning_rate: float = 0.0016
    warmup_steps: int = 3000
    weight_decay: float = 0.1
    max_target_length: int = 30 
    max_prompt_length: int = 15

    save_checkpoints: bool = True
    save_dir: str = "checkpoints"
    seed: int = 0

    log_every_steps: int = 5
    checkpoint_every_steps: int = 20

    emb_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 2048
    batch_size: int = 16384


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConfigRL(Config):
    """Hyperparameter configuration for RL language model training."""
    base_model_path: str = "/mount/llm-post-training-tutorial/checkpoint_post_fix/checkpoint_680"
    num_rollouts: int = 4
    generation_length: int = 9
    temperature = 0.3
    
    learning_rate: float = 5e-5
    warmup_steps: int = 1

    num_train_steps: int = 5