# Multi-property generation configurations

These JSON files define raw target values for CFG-conditioned molecular
generation. Property order in each key must match the order of values in
each `sample_points` entry.

| Property key | Target order |
|---|---|
| `logp+sas` | LogP, SAS |
| `sas+logp` | SAS, LogP |
| `tpsa+logp` | TPSA, LogP |
| `tpsa+sas` | TPSA, SAS |
| `tpsa+logp+sas` | TPSA, LogP, SAS |

`num_samples` specifies samples per target combination. For a grid,
the total requested attempts are the number of target combinations
multiplied by `num_samples`. Generation outputs, logs, and checkpoints
belong under the ignored output/data/checkpoint directories and should
not be committed.
