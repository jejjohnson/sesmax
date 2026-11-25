# Runs described by a spec

Needs the `[sim]` extra (`pip install sesmax-experiment[sim]`). A `RunSpec`, usually a YAML file, says which scenario and model to run and for how long. `simulate`, `spinup` and `restart` find the component that offers the model, run it in chunks and write the artifacts, and the command `sesmax-experiment` does the same from a shell. A run can write checkpoints, and one that a monitor stops resumably is continued by `restart`.

::: sesmax_experiment.sim
    options:
      filters: ["!^app$", "!^main$"]
