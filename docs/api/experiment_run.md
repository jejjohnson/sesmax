# Runs in chunks

A model runs from one time of a `Timeline` to the next, as one compiled integration per chunk, and after each chunk the same stages run on the host: the monitors, a sink for the snapshots, and any [pipekit](https://github.com/jejjohnson/pipekit) operators. Nothing is written unless a stage writes it.

::: sesmax_experiment.run
