# Fenofibrate gene selection and module reduction

This package implements the leakage-controlled representation used by the manuscript:

```text
prepare -> select -> fit-wgcna -> transform -> audit
```

## Publication configuration

- Scope: fenofibrate, rat liver.
- Complete treatments: 10, 100, and 1,000 mg/kg at all eight time points.
- Fit window: 3H, 6H, 9H, 1D, 4D, and 8D only.
- Gene rule: all three replicate signs agree with the non-zero condition median, absolute median log2FC is at least 0.5, and the rule holds in at least half of the 18 fit conditions.
- WGCNA: signed network, biweight midcorrelation, deep split 4, minimum module size 10, merge cut height 0.05, seed 42.
- Projection: gene standardization, PC1 loadings, module standardization, and PC1 orientation are frozen from the 54 fit-window replicate profiles.

The final representation contains 311 selected genes, 293 genes in seven non-grey modules, and 18 grey genes excluded from modeling.

## Commands

Every stage must receive the same organ, compound, run name, split identifier, and WGCNA identity parameters.

```bash
python3 -m src.gene_module_reduction prepare \
  --organ liver --compound fenofibrate --run-name publication
python3 -m src.gene_module_reduction select \
  --organ liver --compound fenofibrate --run-name publication
python3 -m src.gene_module_reduction fit-wgcna \
  --organ liver --compound fenofibrate --run-name publication \
  --deep-split 4 --min-module-size 10 --merge-cut-height 0.05
python3 -m src.gene_module_reduction transform \
  --organ liver --compound fenofibrate --run-name publication \
  --deep-split 4 --min-module-size 10 --merge-cut-height 0.05
```

The compact public reduction in `data/processed/fenofibrate_reduction/` is the frozen downstream input. Raw log2FC matrices are not bundled.

