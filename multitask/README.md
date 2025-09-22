# Multitask PFN Experiments

This module contains lightweight utilities that make it easy to train and benchmark
hierarchical-attention PFNs on synthetic multitask regression data.  The code mirrors
the data generation process used in the multitask PFN project while relying on the
latest TabPFNv2-style architecture that already ships in this repository.

The entry points are:

- `multitask.configs.build_multitask_main_config` – builds a `pfns.train.MainConfig`
  tailored for multitask batches with hierarchical attention enabled.
- `multitask.training_cli` – a small CLI for launching multitask PFN training runs.
- `multitask.benchmark.run_runtime_benchmark` – helper to compare runtime and loss
  across different numbers of tasks using the hierarchical attention flow.

See the tests under `multitask/tests` for examples on how to compose the pieces
programmatically.
