"""Basic residual rollout on five-day module eigengene trajectories.

Mirrors :mod:`src.gene_dynamics.basic_rollout` and
:mod:`src.pathway_dynamics.basic_rollout` on the WGCNA module eigengene
trajectories produced by :mod:`src.gene_module_reduction`. Modules are the
top-level state, so transitions are direct module-to-module residuals.
"""
