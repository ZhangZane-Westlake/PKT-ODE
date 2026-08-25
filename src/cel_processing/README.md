# Open TG-GATEs expression preprocessing

This package builds the rat sample manifest, performs organ-wide RMA with the Brainarray Rat2302_Rn_ENSG version 25 custom CDF, and computes matched-control log2 fold changes.

## Eligibility

- Samples with a serious-pathology flag are excluded before RMA.
- A healthy control has no pathology record and passes all four manuscript biochemical thresholds.
- A treated sample receives log2FC only when at least three healthy controls match organ, time, and administration route.
- RMA is fit once per organ; it is not fit separately by compound or time point.

## Expected raw layout

```text
data/raw/
├── Open-tggates_AllAttribute.tsv
├── open_tggates_pathology.csv
└── ... CEL files ...
```

The Brainarray custom CDF must be installed or made available to R before normalization.

## Commands

```bash
python3 -m src.cel_processing.prepare_rma_samples --dry-run
python3 -m src.cel_processing.prepare_rma_samples
Rscript src/cel_processing/rma_normalize.R --organ liver
python3 -m src.cel_processing.compute_log2fc --organ liver
```

Use `--project-root`, `--raw-dir`, and output arguments when integrating the pipeline elsewhere. Do not replace matched controls with a global control mean.

