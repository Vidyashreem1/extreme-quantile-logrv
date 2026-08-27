# Unconditional LRV Simulation Study

This folder contains the unconditional simulation study comparing extreme-quantile estimators under log-regular variation and ordinary regular variation.

The scripts generate data on the log scale. If `X` is the original variable, the simulation works with

```text
Y = log(X).
```

The target reported by all estimators is

```text
log q_X(1 - alpha) = q_Y(1 - alpha).
```

This avoids numerical overflow from generating `X = exp(Y)` while keeping the comparison on the log-quantile scale used in the paper.

## Main Files

- `simulation_run_logscale.R`: main simulation driver.
- `simulation_run_robustness.R`: robustness study for RV and Log-Weibull
  data-generating models.
- `simulation_project_config.R`: run configuration and environment-variable overrides.
- `simulation_estimators_logscale.R`: Weissman, corrected Weissman, data generators, and error metrics.
- `nn_estimators_logscale.py`: structured neural-network extrapolation estimators.

## Simulation Design

Each replication generates a sample of size `n` from a model specified through the log-scale tail quantile `q_Y`. The default run uses:

```text
n = 500
R = 200
alpha = 1 / (2n)
```

The simulation includes several log-regularly varying models, including Log-Burr, Log-NHW, Log-Fisher, Log-GPD, Log-Inverse-Gamma, and Log-Student cases. The model grid varies the first-order tail index `gamma` and the second-order parameter `rho`.

The intermediate threshold `k` is selected over a grid. In the default configuration,

```text
k_min = 0.03n,
k_max = 0.50n,
k_step = 5.
```

## Robustness Study

The robustness analysis considers Pareto and shifted Burr models under
ordinary regular variation, together with Log-Weibull models having
`beta = 0.75` and `beta = 1.25`. For the Pareto and Log-Weibull
configurations, the corrected estimators use `correction_rho = -1` as a
working tuning value rather than as a model parameter.

## Estimators

The simulation compares six estimators.

- `LRV_Weissman`: Weissman-type estimator on the log-regular variation scale. Since `Y = log(X)` is regularly varying under LRV, this estimator applies Hill extrapolation to `Y`.
- `LRV_Corrected`: bias-corrected Weissman-type estimator on the LRV scale, using a second-order auxiliary correction with the supplied `rho`.
- `NN_LRV_logscale`: structured neural-network estimator trained on LRV double-log spacings.
- `RV_Weissman`: ordinary regular-variation Weissman estimator for `X`, computed directly on the log-quantile scale without forming `X`.
- `RV_Corrected`: ordinary RV bias-corrected Weissman estimator for `X`, also computed on the log-quantile scale.
- `NN_RV_logscale`: structured neural-network estimator trained on ordinary RV log-spacings.

All estimators return an estimate of `log q_X(1 - alpha)`.

## Classical Estimator Scales

For the RV comparators, the code uses the fact that `Y = log(X)`. The ordinary RV Weissman estimator on the original `X` scale satisfies

```text
log qhat_X = Y_{k+1} + gamma_hat_X * log(k / (n alpha)),
```

where `gamma_hat_X` is computed from log-spacings of `X`, equivalently differences of `Y`.

For the LRV estimators, `Y = log(X)` is treated as regularly varying. The Weissman-type estimate is

```text
log qhat_X = qhat_Y
           = Y_{k+1} * (k / (n alpha))^gamma_hat_Y.
```

The corrected LRV estimator multiplies this first-order extrapolation by the second-order correction term.

## Neural-Network Spacing Construction

The neural-network spacing construction below follows the convention with anchor `T_k` and upper ranks `i < k`.

Let `Y_1 >= ... >= Y_n` denote the sorted simulated values.

For the RV neural estimator, the transformed values are

```text
T_i = Y_i.
```

For the LRV neural estimator, the transformed values are

```text
T_i = log(Y_i).
```

For an intermediate rank `k` and upper rank `i < k`, the empirical spacing target is

```text
S(i, k) = T_i - T_k.
```

The neural-network inputs are

```text
x1 = log(k / i),
x2 = log(n / k).
```

The fitted model learns the spacing map

```text
(x1, x2) -> S(i, k).
```

For extrapolation to probability level `alpha`, the fitted spacing is evaluated at

```text
x1 = log(k / (n alpha)),
x2 = log(n / k).
```

The estimated transformed tail value is

```text
T_hat_alpha = T_k + predicted spacing.
```

This is then mapped back to the log-quantile scale. For `NN_RV_logscale`, `T_hat_alpha` is already an estimate of `log q_X`. For `NN_LRV_logscale`, the estimate is

```text
log qhat_X = exp(T_hat_alpha).
```

## Neural-Network Architecture

The neural-network estimator uses a structured one-hidden-layer exponential basis. For inputs `(x1, x2)`, the fitted spacing function is

```text
gamma * x1
+ sum_m w_m * [sigma_e(a_m * x1 + b_m * x2) - sigma_e(c_m * x2)],
```

where

```text
sigma_e(u) = exp(u) - 1.
```

The first-order coefficient `gamma` is constrained to be positive. The slope parameters `a_m`, `b_m`, and `c_m` are constrained to be negative, matching the second- and higher-order tail-expansion structure. The number of paired basis terms is determined by `J`, with `J(J-1)/2` terms.

The candidate grid includes different values of `J`, batch size, and loss function. The selected neural-network estimate is chosen using stability of the extrapolated log-quantile path over the `k` grid.

## Error Metrics

For each replication, the code records the following squared errors:

```text
MedSElog    = (logqhat - logqtrue)^2
MedSEloglog = (log(logqhat) - log(logqtrue))^2
```

The output summaries report medians of these errors over replications.

## Outputs

The main output folder is `simulation_outputs_rv_lrv_comparison/`. It is
created when the simulation is run and is not tracked in the repository.

Outputs from the robustness analysis are saved in `simulation_outputs_rv_logweibull_robustness/`.

Important files include:

- `raw_replications_logscale.csv`: replication-level estimates, selected tuning parameters, and errors.
- `summary_long_logscale.csv`: long-format summary by model and method.
- `wide_MedSElog_logscale.csv`: wide-format median squared error on the log-quantile scale.
- `wide_MedSEloglog_logscale.csv`: wide-format median squared error on the double-log scale.
- `run_metadata_logscale.csv`: run settings.

Model-specific raw files are also written for each simulation design.

The final section of `simulation_run_logscale.R` displays the selected-`k`
boxplot for the Log-Burr configuration with `gamma = 1` and `rho = -1`.

## Reproducing the Simulation

Run the simulation from this folder:

```r
source("simulation_run_logscale.R")
```

Run the RV/Log-Weibull robustness study with:

```r
source("simulation_run_robustness.R")
```

For a small smoke test, override the number of replications and neural-network epochs:

```r
Sys.setenv(SIM_R = "5")
Sys.setenv(SIM_NN_EPOCHS = "5")
Sys.setenv(SIM_PARALLEL_WORKERS = "1")
source("simulation_run_logscale.R")
```

Useful environment variables:

```r
Sys.setenv(SIM_R = "200")
Sys.setenv(SIM_N = "500")
Sys.setenv(SIM_NN_EPOCHS = "50")
Sys.setenv(SIM_J_GRID = "2,3")
Sys.setenv(SIM_NN_BATCH_SIZES = "512")
Sys.setenv(SIM_NN_LOSSES = "l1")
Sys.setenv(SIM_PARALLEL_WORKERS = "4")
Sys.setenv(SIM_PYTHON_THREADS = "1")
```

If needed, set `SIM_PYTHON` to the Python executable for an environment with PyTorch installed.

## Software Requirements

R packages:

- `dplyr`
- `ggplot2`
- `reticulate`

Python packages:

- `numpy`
- `torch`
