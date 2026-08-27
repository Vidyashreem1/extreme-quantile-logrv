from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
if torch_threads > 0:
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(max(1, min(2, torch_threads)))


TAIL_SCALES = ("RV", "LRV")
MODEL_FAMILIES = ("CENN", "LDNN")


class SigmaE(nn.Module):
    """Exponential basis activation sigma_e(x) = exp(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, min=-80.0, max=40.0))


class CoefficientNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = max(8, 2 * input_dim + 10)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CENNNet(nn.Module):
    """Conditional extrapolation network with paired exponential basis terms."""

    def __init__(self, covariate_dim: int, j_order: int):
        super().__init__()
        self.basis_count = max(1, int(j_order) * (int(j_order) - 1) // 2)
        self.coefficients = CoefficientNet(covariate_dim, 1 + 4 * self.basis_count)
        self.sigma_e = SigmaE()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        covariates = x[:, 2:]
        raw = self.coefficients(covariates)
        gamma = torch.nn.functional.softplus(raw[:, 0])
        coef = raw[:, 1:].reshape(-1, self.basis_count, 4)
        w1 = coef[:, :, 0]
        # Enforce negative slope parameters.
        w2 = -torch.nn.functional.softplus(coef[:, :, 1])
        w3 = -torch.nn.functional.softplus(coef[:, :, 2])
        w4 = -torch.nn.functional.softplus(coef[:, :, 3])
        term = self.sigma_e(w2 * x1[:, None] + w3 * x2[:, None]) - self.sigma_e(w4 * x2[:, None])
        return gamma * x1 + torch.sum(w1 * term, dim=1)


class UnconditionalFNet(nn.Module):
    def __init__(self, j_order: int):
        super().__init__()
        self.basis_count = max(1, int(j_order) * (int(j_order) - 1) // 2)
        self.gamma_raw = nn.Parameter(torch.tensor(0.1))
        self.w1 = nn.Parameter(torch.zeros(self.basis_count))
        self.rho2_raw = nn.Parameter(torch.zeros(self.basis_count))
        self.rho3_raw = nn.Parameter(torch.zeros(self.basis_count))
        self.rho4_raw = nn.Parameter(torch.zeros(self.basis_count))
        self.sigma_e = SigmaE()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gamma = torch.nn.functional.softplus(self.gamma_raw)
        rho2 = -torch.nn.functional.softplus(self.rho2_raw)
        rho3 = -torch.nn.functional.softplus(self.rho3_raw)
        rho4 = -torch.nn.functional.softplus(self.rho4_raw)
        term = self.sigma_e(rho2 * x1[:, None] + rho3 * x2[:, None]) - self.sigma_e(rho4 * x2[:, None])
        return gamma * x1 + torch.sum(self.w1 * term, dim=1)


class LDNNNet(nn.Module):
    """Location-dispersion network using the three-order-statistic G formula."""

    def __init__(self, j_order: int):
        super().__init__()
        self.f1 = UnconditionalFNet(j_order)
        self.f2 = UnconditionalFNet(j_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        numerator = torch.exp(torch.clamp(self.f1(x1, x2), min=-40.0, max=40.0)) - 1.0
        denominator = 1.0 - torch.exp(torch.clamp(self.f2(x3, x2), min=-40.0, max=40.0))
        return numerator / torch.clamp(denominator, min=1e-6)


@dataclass
class FittedModel:
    model: nn.Module
    mu: np.ndarray
    sd: np.ndarray
    input_dim: int
    model_family: str
    tail_scale: str
    j_order: int
    batch_size: int
    loss: str


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize(x: np.ndarray):
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd, mu, sd


def transformed_values(y: np.ndarray, tail_scale: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    y = np.sort(y)[::-1]
    if tail_scale == "RV":
        return y
    if tail_scale == "LRV":
        return np.log(y)
    raise ValueError(f"Unknown tail scale: {tail_scale}")


def inverse_transform(t_value: float, tail_scale: str) -> float:
    if not np.isfinite(t_value):
        return np.nan
    if tail_scale == "RV":
        return float(t_value)
    if tail_scale == "LRV":
        if t_value > 709.0:
            return float("inf")
        return float(np.exp(t_value))
    raise ValueError(f"Unknown tail scale: {tail_scale}")


def fit_spacing_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_family: str,
    tail_scale: str,
    j_order: int,
    batch_size: int,
    loss: str,
    epochs: int,
    lr: float,
    seed: int,
):
    if x_train is None or y_train is None or len(y_train) < 20:
        return None

    set_seed(seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    ok = np.all(np.isfinite(x_train), axis=1) & np.isfinite(y_train)
    x_train = x_train[ok]
    y_train = y_train[ok]
    if len(y_train) < 20:
        return None

    x_std, mu, sd = standardize(x_train)
    dataset = TensorDataset(
        torch.tensor(x_std, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

    if model_family == "CENN":
        model = CENNNet(covariate_dim=max(1, x_train.shape[1] - 2), j_order=int(j_order))
    elif model_family == "LDNN":
        model = LDNNNet(j_order=int(j_order))
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.L1Loss() if str(loss).lower() == "l1" else nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss_value = loss_fn(model(xb), yb)
            loss_value.backward()
            opt.step()

    return FittedModel(
        model=model,
        mu=mu,
        sd=sd,
        input_dim=x_train.shape[1],
        model_family=model_family,
        tail_scale=tail_scale,
        j_order=int(j_order),
        batch_size=int(batch_size),
        loss=str(loss),
    )


def predict_spacing(fitted: FittedModel, row: Iterable[float]) -> float:
    row = np.asarray(list(row), dtype=np.float32).reshape(1, -1)
    row = (row - fitted.mu) / fitted.sd
    fitted.model.eval()
    with torch.no_grad():
        return float(fitted.model(torch.tensor(row, dtype=torch.float32)).item())


def make_training_rows(y, covariates, grid_id, k_grid, model_family: str, tail_scale: str, tau_grid=None):
    y = np.asarray(y, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    grid_id = np.asarray(grid_id)
    k_grid = np.asarray(k_grid, dtype=int)
    tau_grid = k_grid if tau_grid is None else np.asarray(tau_grid, dtype=int)
    rows = []
    targets = []

    for gid in np.unique(grid_id):
        idx = np.where(grid_id == gid)[0]
        t_values = transformed_values(y[idx], tail_scale)
        n = t_values.size
        if n < 6:
            continue
        valid_k = k_grid[(k_grid >= 2) & (k_grid < n)]
        if valid_k.size == 0:
            continue
        cov = np.nanmean(covariates[idx, :], axis=0)

        for k in valid_k:
            if model_family == "CENN":
                i = np.arange(1, int(k), dtype=float)
                base = np.column_stack([
                    np.log(k / i),
                    np.full(int(k) - 1, np.log(n / k), dtype=float),
                ])
                cov_block = np.repeat(cov.reshape(1, -1), int(k) - 1, axis=0)
                base = np.column_stack([base, cov_block])
                rows.append(base)
                targets.append(t_values[: int(k) - 1] - t_values[int(k) - 1])
            else:
                j_values = tau_grid[(tau_grid > k) & (tau_grid < n)]
                for j in j_values:
                    i = np.arange(1, int(k), dtype=float)
                    numerator = t_values[: int(k) - 1] - t_values[int(k) - 1]
                    denominator = t_values[int(k) - 1] - t_values[int(j) - 1]
                    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
                        continue
                    base = np.column_stack([
                        np.log(k / i),
                        np.full(int(k) - 1, np.log(n / k), dtype=float),
                        np.full(int(k) - 1, np.log(k / j), dtype=float),
                    ])
                    rows.append(base)
                    targets.append(numerator / denominator)

    if not rows:
        return None, None
    return np.vstack(rows), np.concatenate(targets)


def valid_tau_grid(tau_grid, k_delta: int, n: int):
    tau_grid = np.asarray(tau_grid, dtype=int)
    return tau_grid[(tau_grid > int(k_delta)) & (tau_grid < int(n))]


def estimate_logq(fitted: FittedModel, target_y, target_covariate, k_delta: int, alpha: float, k_tau: int | None = None):
    t_values = transformed_values(np.asarray(target_y, dtype=float), fitted.tail_scale)
    n = t_values.size
    k_delta = int(k_delta)
    if k_delta < 2 or k_delta >= n or alpha <= 0:
        return np.nan

    x1 = float(np.log(k_delta / (n * alpha)))
    x2 = float(np.log(n / k_delta))
    t_delta = float(t_values[k_delta - 1])

    if fitted.model_family == "CENN":
        row = np.concatenate([
            np.asarray([x1, x2], dtype=float),
            np.asarray(target_covariate, dtype=float),
        ])
        t_hat = t_delta + predict_spacing(fitted, row)
        return inverse_transform(t_hat, fitted.tail_scale)

    if k_tau is None:
        return np.nan
    k_tau = int(k_tau)
    if k_tau <= k_delta or k_tau >= n:
        return np.nan
    t_tau = float(t_values[k_tau - 1])
    denom_gap = t_delta - t_tau
    if not np.isfinite(denom_gap) or abs(denom_gap) < 1e-12:
        return np.nan

    x3 = float(np.log(k_delta / k_tau))
    g_ratio = predict_spacing(fitted, [x1, x2, x3])
    if not np.isfinite(g_ratio):
        return np.nan
    t_hat = t_delta + denom_gap * g_ratio
    return inverse_transform(t_hat, fitted.tail_scale)


def anchor_score(fitted: FittedModel, target_y, target_covariate, k_delta: int, anchor_grid, k_tau: int | None = None):
    y_sorted = np.sort(np.asarray(target_y, dtype=float)[np.isfinite(target_y) & (np.asarray(target_y, dtype=float) > 0)])[::-1]
    n = y_sorted.size
    errors = []
    for anchor_rank in np.asarray(anchor_grid, dtype=int):
        if anchor_rank < 1 or anchor_rank >= min(k_delta, n):
            continue
        alpha_anchor = anchor_rank / n
        observed = float(y_sorted[anchor_rank - 1])
        if fitted.model_family == "LDNN":
            if k_tau is None or k_tau <= k_delta or k_tau >= n:
                continue
        pred = estimate_logq(fitted, target_y, target_covariate, k_delta, alpha_anchor, k_tau=k_tau)
        if np.isfinite(pred) and np.isfinite(observed):
            errors.append(abs(pred - observed))
    if not errors:
        return np.inf, 0
    return float(np.median(errors)), int(len(errors))


def run_framework(
    y,
    covariates,
    grid_id,
    target_grid_id,
    target_covariate,
    heldout_log_max,
    alpha_test,
    k_grid,
    anchor_grid,
    j_grid,
    batch_sizes,
    losses,
    epochs,
    lr,
    seed,
    tail_scales=None,
    tau_grid=None,
):
    y = np.asarray(y, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    grid_id = np.asarray(grid_id)
    # The R driver encodes each Mahalanobis neighbourhood as one training
    # sample, identified by the target grid id. This lets CENN/LDNN learn a
    # global extrapolation over all sampled neighbourhoods while estimating
    # the intermediate anchor from the selected target neighbourhood.
    target_y = y[grid_id == target_grid_id]
    target_covariate = np.asarray(target_covariate, dtype=float)
    if tau_grid is None:
        tau_grid = np.asarray(k_grid, dtype=int)
    tau_grid = np.asarray(tau_grid, dtype=int)
    rows = []
    selected_rows = []
    if tail_scales is None:
        tail_scales = ("LRV",)
    tail_scales = tuple(str(x).upper() for x in tail_scales)

    for tail_scale in tail_scales:
        if tail_scale not in TAIL_SCALES:
            raise ValueError(f"Unknown tail scale: {tail_scale}")
        for model_family in MODEL_FAMILIES:
            x_train, y_train = make_training_rows(
                y=y,
                covariates=covariates,
                grid_id=grid_id,
                k_grid=k_grid,
                model_family=model_family,
                tail_scale=tail_scale,
                tau_grid=tau_grid,
            )
            if x_train is None:
                continue

            best = None
            for j_order in j_grid:
                for batch_size in batch_sizes:
                    for loss in losses:
                        fitted = fit_spacing_model(
                            x_train=x_train,
                            y_train=y_train,
                            model_family=model_family,
                            tail_scale=tail_scale,
                            j_order=int(j_order),
                            batch_size=int(batch_size),
                            loss=str(loss),
                            epochs=int(epochs),
                            lr=float(lr),
                            seed=int(seed),
                        )
                        if fitted is None:
                            continue

                        for k_delta in np.asarray(k_grid, dtype=int):
                            if k_delta < 2 or k_delta >= len(target_y):
                                continue
                            tau_candidates = [0] if model_family == "CENN" else valid_tau_grid(tau_grid, int(k_delta), len(target_y))
                            for k_tau in tau_candidates:
                                k_tau_arg = int(k_tau) if model_family == "LDNN" else None
                                mad, n_anchor = anchor_score(
                                    fitted=fitted,
                                    target_y=target_y,
                                    target_covariate=target_covariate,
                                    k_delta=int(k_delta),
                                    anchor_grid=anchor_grid,
                                    k_tau=k_tau_arg,
                                )
                                logqhat = estimate_logq(
                                    fitted=fitted,
                                    target_y=target_y,
                                    target_covariate=target_covariate,
                                    k_delta=int(k_delta),
                                    alpha=float(alpha_test),
                                    k_tau=k_tau_arg,
                                )
                                mse_log = (logqhat - heldout_log_max) ** 2 if np.isfinite(logqhat) else np.nan
                                mse_loglog = (
                                    (np.log(logqhat) - np.log(heldout_log_max)) ** 2
                                    if np.isfinite(logqhat) and logqhat > 0 and heldout_log_max > 0
                                    else np.nan
                                )
                                row = {
                                    "model": f"{model_family}-{tail_scale}",
                                    "model_family": model_family,
                                    "tail_scale": tail_scale,
                                    "selected_by": "minimum_anchor_MAD",
                                    "anchor_mad": mad,
                                    "anchor_count": n_anchor,
                                    "k_delta": int(k_delta),
                                    "k_tau": int(k_tau),
                                    "J": int(j_order),
                                    "batch_size": int(batch_size),
                                    "loss": str(loss),
                                    "alpha_test": float(alpha_test),
                                    "heldout_log_max": float(heldout_log_max),
                                    "logqhat": float(logqhat) if np.isfinite(logqhat) else np.nan,
                                    "MSE_log": float(mse_log) if np.isfinite(mse_log) else np.nan,
                                    "MSE_loglog": float(mse_loglog) if np.isfinite(mse_loglog) else np.nan,
                                }
                                rows.append(row)
                                if best is None or row["anchor_mad"] < best["anchor_mad"]:
                                    best = dict(row)

            if best is not None:
                selected_rows.append(best)

    return rows, selected_rows


def split_ints(value: str):
    return [int(x) for x in str(value).split(",") if str(x).strip()]


def split_floats(value: str):
    return [float(x) for x in str(value).split(",") if str(x).strip()]


def split_strings(value: str):
    return [str(x).strip() for x in str(value).split(",") if str(x).strip()]


def read_input_csv(path):
    y = []
    covariates = []
    grid_id = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            y.append(float(row["Y"]))
            covariates.append([
                float(row["lat_std"]),
                float(row["lon_std"]),
                float(row["alt_mean_std"]),
            ])
            grid_id.append(row["grid_id"])
    return np.asarray(y, dtype=float), np.asarray(covariates, dtype=float), np.asarray(grid_id)


def write_csv(path, rows):
    fieldnames = [
        "model", "model_family", "tail_scale", "selected_by", "anchor_mad", "anchor_count",
        "k_delta", "k_tau", "J", "batch_size", "loss", "alpha_test", "heldout_log_max",
        "logqhat", "MSE_log", "MSE_loglog",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Conditional CENN/LDNN log-regular framework for log-rainfall")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--candidate-output-csv", required=True)
    parser.add_argument("--selected-output-csv", required=True)
    parser.add_argument("--target-grid-id", required=True)
    parser.add_argument("--target-covariate", required=True)
    parser.add_argument("--heldout-log-max", type=float, required=True)
    parser.add_argument("--alpha-test", type=float, required=True)
    parser.add_argument("--k-grid", required=True)
    parser.add_argument("--tau-grid", required=True)
    parser.add_argument("--anchor-grid", required=True)
    parser.add_argument("--j-grid", default="2,3")
    parser.add_argument("--batch-sizes", default="512")
    parser.add_argument("--losses", default="l1")
    parser.add_argument("--tail-scales", default="LRV")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    y, covariates, grid_id = read_input_csv(args.input_csv)
    candidates, selected = run_framework(
        y=y,
        covariates=covariates,
        grid_id=grid_id,
        target_grid_id=args.target_grid_id,
        target_covariate=split_floats(args.target_covariate),
        heldout_log_max=float(args.heldout_log_max),
        alpha_test=float(args.alpha_test),
        k_grid=split_ints(args.k_grid),
        tau_grid=split_ints(args.tau_grid),
        anchor_grid=split_ints(args.anchor_grid),
        j_grid=split_ints(args.j_grid),
        batch_sizes=split_ints(args.batch_sizes),
        losses=split_strings(args.losses),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
        tail_scales=split_strings(args.tail_scales),
    )
    write_csv(args.candidate_output_csv, candidates)
    write_csv(args.selected_output_csv, selected)


if __name__ == "__main__":
    main()
