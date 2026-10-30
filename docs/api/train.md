# Training rollouts

Step a model many times inside one `jax.lax.scan` and score the trajectory against a reference, so learned parameters and closures train by differentiating through a window.

::: sesmax_core.train
