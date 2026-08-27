# Conditional CENN/LDNN Log-Rainfall Analysis

This folder contains the conditional extreme rainfall analysis for the Western Ghats application. The pipeline estimates extreme conditional rainfall levels using neural-network extrapolation under log-regular variation, with ordinary regular-variation versions included as sensitivity comparators.

The main scripts are:

- `07.Conditional_CENN_LDNN_LogRainfall.R`: prepares target neighbourhoods, builds training panels, calls the Python neural-network engine, and aggregates candidate and selected models.
- `cenn_ldnn_lograinfall_engine.py`: fits the conditional extrapolation neural networks and writes candidate and selected estimates.
- `08.CENN_LDNN_LogRainfall_Diagnostics.R`: creates diagnostic summaries and figures from the fitted output.

## Data Layout

The analysis uses gridded daily rainfall data and grid-level covariates.

- `data/western_ghats_coordinates.csv`: grid coordinates and altitude summaries.
- `data/Western ghats data/`: daily rainfall files for individual grid cells.
- `outputs/cenn_ldnn_lograinfall/`: generated output folder for the fitted
  CENN/LDNN analysis.

The raw rainfall files are not included in the repository. Their locations
can instead be supplied through the environment variables described below.

## Statistical Design

- Response: `Y = log(Precipitation)`.
- Covariates: standardized latitude, longitude, and mean altitude.
- Target grids: stratified sample of grids from three altitude regions:
  - coastal: `alt_mean < 100`
  - mid_range: `100 <= alt_mean < 600`
  - hilly: `alt_mean >= 600`
- Neighbourhoods: for each target grid, the pooled sample is formed using Mahalanobis nearest neighbours in `(longitude, latitude, alt_mean)`.
- Holdout value: the maximum positive rainfall value in the target neighbourhood is removed before fitting and retained as the local held-out extreme.
- Training sample: the remaining positive rainfall observations define the
  local sample and are transformed to `Y = log(Precipitation)`. Observations
  with rainfall exceeding 1 mm enter the spacing construction, ensuring that
  `Y > 0` and that `log(Y)` is defined. Each observation is stored together
  with the standardized covariates of the target grid.

The reported run uses 30 sampled target grids per altitude region and 25 neighbours per target grid. These choices can be changed using environment variables.

## Tail Scales

The implementation supports two tail scales.

- `LRV`: log-regular variation scale. The network is trained on double-log spacings using `T = log(Y)`, where `Y = log(Precipitation)`.
- `RV`: ordinary regular-variation comparator scale. The network is trained on log-rainfall spacings using `T = Y`.

For CENN, the LRV scale is the proposed specification and the RV scale is a
comparator. The location-dispersion construction is formulated for
`Y = log(Precipitation)`, so LDNN-RV corresponds to that construction and
LDNN-LRV is an additional scale-sensitivity specification.

## CENN Spacing Construction

For a target neighbourhood, let `T_1 >= T_2 >= ... >= T_n` denote the sorted transformed values on the chosen tail scale. For an intermediate rank `k` and upper rank `i < k`, the empirical conditional spacing target is

```text
S(i, k) = T_i - T_k.
```

The corresponding inputs are

```text
x1 = log(k / i),
x2 = log(n / k),
covariates = (lat_std, lon_std, alt_mean_std).
```

The CENN model learns the map

```text
(x1, x2, covariates) -> S(i, k).
```

For extrapolation to tail probability `alpha`, the model predicts the spacing from the intermediate rank `k_delta` to the target level using

```text
x1 = log(k_delta / (n * alpha)),
x2 = log(n / k_delta).
```

The predicted transformed tail value is

```text
T_hat_alpha = T_k_delta + predicted spacing.
```

This value is then mapped back to the log-rainfall scale. Under `LRV`, this means applying `exp(T_hat_alpha)`, because `T = log(Y)`.

## CENN Neural Architecture

The CENN network uses a one-hidden-layer exponential-basis form designed for the log-spacing expansion. The covariates are passed through a small coefficient network. This coefficient network outputs:

- a positive first-order tail coefficient `gamma(y)`;
- weights for paired exponential basis terms;
- negative slope parameters for the `x1` and `x2` directions.

For input `(x1, x2, y)`, the fitted spacing has the form

```text
gamma(y) * x1
+ sum_m w_m(y) * [exp(a_m(y) * x1 + b_m(y) * x2) - exp(c_m(y) * x2)].
```

The parameters `gamma(y)`, `w_m(y)`, `a_m(y)`, `b_m(y)`, and `c_m(y)` are learned from the pooled target-neighbourhood training panel. The number of paired basis terms is determined by `J`, with `J(J-1)/2` terms.

## LDNN Spacing Construction

The LDNN model uses two intermediate anchors. For transformed order statistics `T_1 >= ... >= T_n`, choose ranks `i < k < j`. The training target is the ratio

```text
G(i, k, j) = (T_i - T_k) / (T_k - T_j).
```

The corresponding inputs are

```text
x1 = log(k / i),
x2 = log(n / k),
x3 = log(k / j).
```

By default, the second anchor `j` is selected from
`{300, 500, 750, 1000, 1500, 2000}`, supplemented by `2k`, `floor(n/2)`, and
`n-1` whenever `k < j < n`. The fixed grid can be changed through
`CENN_LDNN_TAU_GRID`.

For extrapolation to tail probability `alpha`, LDNN predicts a ratio

```text
G_hat = G(x1, x2, x3),
```

and then estimates

```text
T_hat_alpha = T_k_delta + (T_k_delta - T_k_tau) * G_hat.
```

Covariates enter LDNN through the intermediate conditional anchors rather than through the extrapolation network itself.

## LDNN Neural Architecture

LDNN uses an unconditional exponential-basis extrapolation network for the ratio map. Internally, it fits paired exponential spacing functions and combines them as

```text
G_hat(x1, x2, x3)
= [exp(f_1(x1, x2)) - 1] / [1 - exp(f_2(x3, x2))],
```

where each `f` has the same structured exponential-basis form used for log-spacing extrapolation.

## Model Selection

For each candidate model, the fitted extrapolation rule is tested against held-in anchor order statistics. The score is the median absolute deviation between predicted and observed held-in anchors. The selected model is the candidate with the smallest anchor MAD.

Candidate grids include:

- `k_delta`: intermediate rank for the main anchor;
- `k_tau`: second anchor rank for LDNN;
- `J`: number of expansion orders used to determine the paired-basis count;
- batch size;
- loss function;
- tail scale.

## Outputs

The main output folder is `outputs/cenn_ldnn_lograinfall/`. It is generated
when the analysis is run and is not tracked in the repository.

Important files include:

- `sampled_target_grids.csv`: target grids used in the analysis.
- `run_metadata.csv`: run settings and tuning grids.
- `inputs/global_m*_input.csv`: pooled neural-network training panel.
- `global_m*_target_metadata.csv`: metadata for sampled target neighbourhoods.
- `all_candidate_models.csv`: all fitted candidate models and validation scores.
- `all_selected_models.csv`: selected model for each target and method.
- `summary_by_region_neighbor_model.csv`: summary metrics by altitude region, neighbour count, and model.
- `overall_summary.csv`: overall model comparison.

Diagnostic figures are saved as PDF files directly in the main output folder.

## Reproducing the Analysis

Run the main analysis from this folder:

```r
source("07.Conditional_CENN_LDNN_LogRainfall.R")
```

Then generate diagnostics:

```r
source("08.CENN_LDNN_LogRainfall_Diagnostics.R")
```

For a small smoke test, reduce the run size before sourcing the main script:

```r
Sys.setenv(CENN_LDNN_SAMPLE_PER_REGION = "1")
Sys.setenv(CENN_LDNN_M_NEIGHBORS = "10")
Sys.setenv(CENN_LDNN_K_GRID = "25,50")
Sys.setenv(CENN_LDNN_TAU_GRID = "75,100")
Sys.setenv(CENN_LDNN_EPOCHS = "2")
source("07.Conditional_CENN_LDNN_LogRainfall.R")
```

Useful environment variables:

```r
Sys.setenv(CENN_LDNN_PYTHON = "path/to/python")
Sys.setenv(CENN_LDNN_COORD_FILE = "path/to/western_ghats_coordinates.csv")
Sys.setenv(CENN_LDNN_RAIN_DIR = "path/to/grid-level/rainfall/files")
Sys.setenv(CENN_LDNN_OUT_DIR = "path/to/output/directory")
Sys.setenv(CENN_LDNN_SAMPLE_PER_REGION = "30")
Sys.setenv(CENN_LDNN_M_NEIGHBORS = "25")
Sys.setenv(CENN_LDNN_K_GRID = "25,50,75,100,150,200")
Sys.setenv(CENN_LDNN_TAU_GRID = "300,500,750,1000,1500,2000")
Sys.setenv(CENN_LDNN_ANCHOR_GRID = "1,2,3,5,10")
Sys.setenv(CENN_LDNN_J_GRID = "2,3")
Sys.setenv(CENN_LDNN_BATCH_SIZES = "512")
Sys.setenv(CENN_LDNN_LOSSES = "l1")
Sys.setenv(CENN_LDNN_TAIL_SCALES = "LRV,RV")
Sys.setenv(CENN_LDNN_EPOCHS = "50")
Sys.setenv(CENN_LDNN_PARALLEL = "TRUE")
Sys.setenv(CENN_LDNN_WORKERS = "4")
Sys.setenv(CENN_LDNN_PYTHON_THREADS = "2")
```

## Software Requirements

R packages:

- `data.table`
- `ggnewscale`
- `ggplot2`
- `processx`
- `scales`
- `viridis`

Python packages:

- `numpy`
- `torch`

The Python executable used by the R driver should point to an environment where PyTorch is installed.
