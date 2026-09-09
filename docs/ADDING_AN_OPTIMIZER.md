# Adding an optimizer

Add the algorithm implementation under `src/optim/`, add one explicit name branch in `src/optim/lowp_adapter.py`, and add the name to `src/config/recipe.py`. Reuse `train_platform.parameter_groups`; tied embedding/head storage must appear once.

The optimizer must preserve `state_dict` exactly, expose all numerically relevant settings in the recipe/source metadata, reject unsupported device/API semantics, and work with `grad=None`. Add focused formula, multi-step, state dtype/size, save/load and device tests. For a paired experiment, copy the full recipe and vary only the optimizer fields declared to `compare_runs.py`; never add an implicit fallback or dynamic plugin registry.
