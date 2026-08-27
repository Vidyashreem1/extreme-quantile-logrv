from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def sigma_e(x: torch.Tensor) -> torch.Tensor:
    """Exponential basis function sigma_e(x) = exp(x) - 1."""
    return torch.expm1(torch.clamp(x, min=-50.0, max=50.0))


class StructuredExtrapolationNet(nn.Module):
    """Structured one-hidden-layer extrapolation network."""

    def __init__(self, input_dim: int = 2, j_order: int = 2):
        super().__init__()
        if int(input_dim) != 2:
            raise ValueError("StructuredExtrapolationNet expects two inputs: x1 and x2.")
        hidden_dim = max(1, int(j_order) * (int(j_order) - 1) // 2)
        self.raw_gamma = nn.Parameter(torch.tensor(-2.25, dtype=torch.float32))
        self.raw_w1 = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        self.raw_w2 = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        self.raw_w3 = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        self.raw_w4 = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.raw_w1, mean=0.00, std=0.05)
        nn.init.normal_(self.raw_w2, mean=-2.25, std=0.10)
        nn.init.normal_(self.raw_w3, mean=-2.25, std=0.10)
        nn.init.normal_(self.raw_w4, mean=-2.25, std=0.10)

    def constrained_parameters(self):
        return {
            "gamma": torch.nn.functional.softplus(self.raw_gamma),
            "w1": self.raw_w1,
            "w2": -torch.nn.functional.softplus(self.raw_w2),
            "w3": -torch.nn.functional.softplus(self.raw_w3),
            "w4": -torch.nn.functional.softplus(self.raw_w4),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        params = self.constrained_parameters()
        gamma = params["gamma"]
        w1 = params["w1"]
        w2 = params["w2"]
        w3 = params["w3"]
        w4 = params["w4"]
        curved = sigma_e(x1 * w2 + x2 * w3) - sigma_e(x2 * w4)
        return gamma * x[:, 0] + torch.sum(curved * w1, dim=1)


@dataclass
class FittedSpacingModel:
    model: StructuredExtrapolationNet
    mu: np.ndarray
    sd: np.ndarray
    scale: str
    j_order: int
    batch_size: int
    loss: str


def _set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _use_extrapolation_coordinates(x: np.ndarray):
    mu = np.zeros(x.shape[1], dtype=np.float32)
    sd = np.ones(x.shape[1], dtype=np.float32)
    return x, mu, sd


def _sorted_positive(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    return np.sort(y)[::-1]


def _make_spacing_data(y, scale: str, max_k: int | None = None):
    """Build empirical spacing targets from Y = log X.

    scale="rv_logq": RV estimator on X, computed from Y=log X without forming
    X. Its log-spacings are Y_i - Y_k, and the output is log qhat_X.

    scale="lrv_logq": LRV estimator for X, which treats Y = log X as RV. Its
    log-spacings are log(Y_i) - log(Y_k), and the output is qhat_Y = log qhat_X.
    """
    y = _sorted_positive(y)
    n = y.size
    if n < 4:
        return None, None, y
    if max_k is None:
        max_k = n - 1
    max_k = int(min(max_k, n - 1))
    if max_k < 3:
        return None, None, y

    if scale == "rv_logq":
        values = y
    elif scale == "lrv_logq":
        values = np.log(np.maximum(y, np.finfo(float).tiny))
    else:
        raise ValueError("scale must be 'rv_logq' or 'lrv_logq'")

    rows = []
    targets = []
    for k in range(2, max_k + 1):
        i = np.arange(1, k, dtype=float)
        rows.append(np.column_stack([
            np.log(k / i),
            np.full(k - 1, np.log(n / k), dtype=float),
        ]))
        targets.append(values[: k - 1] - values[k - 1])

    return np.vstack(rows), np.concatenate(targets), y


def _fit_spacing_model(
    y,
    scale: str,
    j_order: int = 2,
    batch_size: int = 256,
    loss: str = "l1",
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 123,
    max_k: int | None = None,
):
    _set_seed(seed)
    x_train, y_train, y_sorted = _make_spacing_data(y, scale=scale, max_k=max_k)
    if x_train is None:
        return None, y_sorted

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    x_std, mu, sd = _use_extrapolation_coordinates(x_train)

    dataset = TensorDataset(
        torch.tensor(x_std, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

    model = StructuredExtrapolationNet(input_dim=2, j_order=int(j_order))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.L1Loss() if loss == "l1" else nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss_value = loss_fn(pred, yb)
            loss_value.backward()
            opt.step()

    return (
        FittedSpacingModel(
            model=model,
            mu=mu,
            sd=sd,
            scale=scale,
            j_order=int(j_order),
            batch_size=int(batch_size),
            loss=loss,
        ),
        y_sorted,
    )


def _predict_spacing(fitted: FittedSpacingModel, x1: float, x2: float) -> float:
    row = np.asarray([[x1, x2]], dtype=np.float32)
    row = (row - fitted.mu) / fitted.sd
    fitted.model.eval()
    with torch.no_grad():
        return float(fitted.model(torch.tensor(row, dtype=torch.float32)).item())


def _estimate_logq_from_model(fitted: FittedSpacingModel, y_sorted, k: int, alpha: float) -> float:
    n = y_sorted.size
    k = int(k)
    if k < 2 or k >= n or alpha <= 0:
        return float("nan")

    anchor = float(y_sorted[k - 1])
    if not np.isfinite(anchor) or anchor <= 0:
        return float("nan")

    pred = _predict_spacing(
        fitted,
        x1=float(np.log(k / (n * alpha))),
        x2=float(np.log(n / k)),
    )

    if fitted.scale == "rv_logq":
        return float(anchor + pred)

    log_logq = np.log(anchor) + pred
   # Preserve overflow as Inf rather than truncating the estimate.
    return float(np.exp(log_logq))


def _mad(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("inf")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _select_k_by_tree(values, k_grid):
    values = np.asarray(values, dtype=float)
    k_grid = np.asarray(k_grid, dtype=int)
    ok = np.isfinite(values)
    values = values[ok]
    k_grid = k_grid[ok]
    if values.size == 0:
        return int(k_grid[0]) if k_grid.size else 2
    if values.size < 4:
        return int(k_grid[int(values.size // 2)])

    left = 0
    right = values.size - 1
    while (right - left) >= 3:
        mid = (left + right) // 2
        v_left = np.var(values[left : mid + 1])
        v_right = np.var(values[mid : right + 1])
        if v_left <= v_right:
            right = mid
        else:
            left = mid
    return int(k_grid[(left + right) // 2])


def nn_selected_logq_estimator(
    y,
    alpha: float,
    scale: str,
    k_grid=None,
    j_grid=None,
    batch_sizes=None,
    losses=None,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 123,
):
    if k_grid is None:
        k_grid = np.arange(15, 376, 5, dtype=int)
    else:
        k_grid = np.atleast_1d(np.asarray(k_grid, dtype=int))
    if j_grid is None:
        j_grid = np.arange(2, 6, dtype=int)
    else:
        j_grid = np.atleast_1d(np.asarray(j_grid, dtype=int))
    if batch_sizes is None:
        batch_sizes = np.asarray([256], dtype=int)
    else:
        batch_sizes = np.atleast_1d(np.asarray(batch_sizes, dtype=int))
    if losses is None:
        losses = ["l1"]
    elif isinstance(losses, str):
        losses = [losses]
    else:
        losses = list(np.atleast_1d(losses))

    best = {
        "logqhat": float("nan"),
        "selected_k": int(k_grid[0]),
        "selected_J": int(j_grid[0]),
        "selected_batch_size": int(batch_sizes[0]),
        "selected_loss": str(losses[0]),
        "mad": float("inf"),
        "training_scale": str(scale),
    }

    for j_order in j_grid:
        for batch_size in batch_sizes:
            for loss in losses:
                fitted, y_sorted = _fit_spacing_model(
                    y,
                    scale=scale,
                    j_order=int(j_order),
                    batch_size=int(batch_size),
                    loss=str(loss),
                    epochs=int(epochs),
                    lr=float(lr),
                    seed=int(seed),
                    max_k=int(np.max(k_grid)),
                )
                if fitted is None:
                    continue

                logq_path = np.asarray([
                    _estimate_logq_from_model(fitted, y_sorted, int(k), float(alpha))
                    for k in k_grid
                ])
                score = _mad(logq_path)
                if score < best["mad"]:
                    selected_k = _select_k_by_tree(logq_path, k_grid)
                    logqhat = _estimate_logq_from_model(fitted, y_sorted, selected_k, float(alpha))
                    best.update({
                        "logqhat": float(logqhat),
                        "selected_k": int(selected_k),
                        "selected_J": int(j_order),
                        "selected_batch_size": int(batch_size),
                        "selected_loss": str(loss),
                        "mad": float(score),
                        "training_scale": str(scale),
                    })

    return best


def nn_rv_logq_estimator(y, alpha, k_grid, j_grid, batch_sizes, losses, epochs=200, lr=0.01, seed=123):
    return nn_selected_logq_estimator(
        y, alpha, "rv_logq", k_grid, j_grid, batch_sizes, losses, epochs, lr, seed
    )


def nn_lrv_logq_estimator(y, alpha, k_grid, j_grid, batch_sizes, losses, epochs=200, lr=0.01, seed=123):
    return nn_selected_logq_estimator(
        y, alpha, "lrv_logq", k_grid, j_grid, batch_sizes, losses, epochs, lr, seed
    )
