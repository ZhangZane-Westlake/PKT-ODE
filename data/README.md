# Data snapshot

## `processed/`

- `fenofibrate_reduction/projection/treatment_module_trajectories.npz`: three doses by eight times by three replicates by seven modules, plus condition means and standard deviations.
- `fenofibrate_reduction/projection/module_bundle.npz`: frozen gene and module standardization plus fixed PC1 loadings.
- `fenofibrate_reduction/wgcna/`: selected module membership and WGCNA configuration diagnostics.
- `pkt_ode_input.npz`: 72 aligned raw and fit-window MAD-normalized module profiles, with sample, regimen, split, and normalization metadata used by the PKT-ODE analysis.
- `published_parameters.tsv`: rounded per-module PKT-ODE parameters reported with the manuscript.
- `benchmark_metrics.tsv`: manuscript day-29 benchmark values.

## `reference/`

- `ppara_targets.tsv`: 58-gene literature-curated PPAR-alpha target snapshot.
- `ppara_module_hits.tsv`: the 27 target genes present in the seven modeled modules.

The derived expression artifacts originate from Open TG-GATEs and retain the source database's attribution and share-alike requirements. Raw CEL files are not redistributed here.

`MANIFEST.tsv` records SHA-256 digests for every machine-readable data artifact so the publication snapshot can be checked independently of local paths.
