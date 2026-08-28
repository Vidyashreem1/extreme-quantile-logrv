# Extreme Quantile Extrapolation under Log-Regular Variation

This repository contains the code accompanying the manuscript on extreme
quantile extrapolation for log-regularly varying tails. It includes the
unconditional simulation study and the conditional Western Ghats rainfall
analysis.

The accompanying manuscript is authored by Vidyashree M and Sreenivasan Ravi.

## Repository Structure

- `simulations/`: RV/LRV simulation drivers, classical estimators, and the
  structured neural-network implementation.
- `rainfall-analysis/`: conditional CENN/LDNN analysis and diagnostic scripts.

Each folder contains a separate README describing the statistical design,
configuration, and execution steps.

## Software Requirements

The analyses use R and Python. The principal R packages are:

- `data.table`
- `dplyr`
- `ggplot2`
- `ggnewscale`
- `processx`
- `reticulate`
- `scales`

The Python dependencies are listed in `requirements.txt`.

## Reproducibility

The default parameter settings and random seeds used for the manuscript are
defined in the scripts. Output directories are created when the analyses are
run and are intentionally excluded from version control.

The raw rainfall data are not redistributed in this repository. Instructions
for configuring the required input locations are provided in
`rainfall-analysis/README.md`.

## License

The source code is released under the MIT License. See `LICENSE`.
