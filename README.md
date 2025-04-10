
## Coming Soon
Expect a readme detailing the layout of the code as well pitfalls for extending soon.

For now the python files that are execution entry points are

`motivation/jax_countdown/data/generate_data.py`
`motivation/jax_countdown/training/train.py`
`motivation/jax_countdown/training/train_rl.py`
`motivation/jax_countdown/training/eval_reward.py`

Some bash scripts that show examples of what arguments these files take are in `motivation/jax_countdown/data/` and `motivation/jax_countdown/training`

## P.S

In `motivation/jax_countdown/training/config.py` the maximum sequence length for the transformer is set to 30 tokens in the line
```
    max_target_length: int = 30 
```
This will cause mysterious errors if you try doing forward passes on generations that are longer than 30. Make sure to change this if you need to work with longer sequences.
