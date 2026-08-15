# Tabular model implementations

This directory contains the code used for the following tabular baselines:

| Model | Training wrapper | Core implementation |
| --- | --- | --- |
| TabM | `tabular_models/TALENT/model/methods/tabm.py` | `tabular_models/TALENT/model/models/tabm.py`, `tabular_models/TALENT/model/lib/tabm/` |
| TabPFN | `tabular_models/TALENT/model/methods/tabpfn.py` | `tabular_models/TALENT/model/models/tabpfn.py`, `tabular_models/TALENT/model/lib/tabpfn/` |
| TabICL | `tabular_models/TALENT/model/methods/tabicl.py` | `tabular_models/TALENT/model/lib/tabicl/` |
| TabTransformer | `tabular_models/TALENT/model/methods/tabtransformer.py` | `tabular_models/TALENT/model/models/tabtransformer.py` |
| FTTransformer | `tabular_models/TALENT/model/methods/ftt.py` | `tabular_models/TALENT/model/models/ftt.py` |
| AMFormer | `tabular_models/TALENT/model/methods/amformer.py` | `tabular_models/TALENT/model/models/amformer.py`, `tabular_models/TALENT/model/lib/amformer/` |
| xRFM | `tabular_models/TALENT/model/classical_methods/xrfm.py` | `tabular_models/xrfm/` |

Only source code and license files are included. Experimental results, datasets, logs, checkpoints, caches, and generated artifacts are excluded. Python comments and standalone documentation strings were removed from the copied sources as requested.

The TALENT-derived files retain the upstream license in `tabular_models/LICENSE`. The bundled xRFM source retains its license in `tabular_models/xrfm-0.2.0.dist-info/licenses/LICENSE`.

## Validation

All included Python files pass `python -m compileall` after comment removal.
