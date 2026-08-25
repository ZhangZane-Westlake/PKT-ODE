# Results snapshot

`figures/` contains the four main figures used in the manuscript. Regenerated figures are written to `reproduced_figures/` by:

```bash
python3 -m src.pkt_ode figures
```

Fresh model fits and metric tables should be written under `reproduction/`. Both generated directories are ignored so that reruns do not overwrite or mix with the publication snapshot.

