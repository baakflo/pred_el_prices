"""Quantile neural network: one small MLP forecasting 99 percentiles for all 24 hours.

Day-level design, as LEAR: one sample per UTC day, inputs from models.lear.build_xy
(price lags d-1, d-2, d-3, d-7 and exogenous lags d, d-1, d-7, gate-safe) plus
day-level extras (fuel settlements, lagged in the dataset), all InvariantScaler
(median/MAD + asinh) scaled on the training window; day-of-week dummies unscaled.

Output head: for every hour, the median plus softplus steps cumulated outwards
(49 down, 49 up), so the 99 percentiles never cross. Loss: pinball averaged
over percentiles and hours, in the scaled target space. The target scaler is a
monotone map per hour, and quantiles commute with monotone maps, so the inverse
transform of a scaled percentile is the same percentile in EUR/MWh.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from pred_el_prices.models.lear import InvariantScaler

QUANTILES = np.arange(1, 100) / 100.0


@dataclass
class QNNConfig:
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 400
    patience: int = 30
    val_share: float = 0.15


class QuantileMLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float, n_q: int = len(QUANTILES)):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ELU(), nn.Dropout(dropout)]
            width = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 24 * n_q)
        self.n_q = n_q
        self.mid = n_q // 2
        # start with a sane spread: softplus(-3) ~ 0.05 per step, i.e. roughly
        # +-2.4 around the median at the 1st/99th percentile in scaled units
        # (softplus(0) per step would span ~+-34, which sinh maps to 1e14 EUR)
        with torch.no_grad():
            bias = self.head.bias.view(24, n_q)
            bias.fill_(-3.0)
            bias[:, self.mid] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Median plus cumulated positive steps outwards on both sides: never crosses."""
        raw = self.head(self.body(x)).view(-1, 24, self.n_q)
        median = raw[..., self.mid : self.mid + 1]
        steps = nn.functional.softplus(raw)
        below = median - torch.flip(torch.cumsum(torch.flip(steps[..., : self.mid], [-1]), -1), [-1])
        above = median + torch.cumsum(steps[..., self.mid + 1 :], -1)
        return torch.cat([below, median, above], dim=-1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss; pred (n, 24, n_q), y (n, 24), q (n_q,)."""
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    n_unscaled: int,
    config: QNNConfig,
    seed: int,
) -> np.ndarray:
    """Fit on (x_train, y_train) and return percentiles for x_pred, (n_pred, 24, 99) EUR/MWh.

    The last `n_unscaled` columns of x (dummies) bypass scaling. Early stopping
    on a random `val_share` of the training days restores the best epoch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    scaler_x = InvariantScaler().fit(x_train[:, :-n_unscaled])
    scaler_y = InvariantScaler().fit(y_train)

    def _x(x):
        return np.hstack([scaler_x.transform(x[:, :-n_unscaled]), x[:, -n_unscaled:]])

    xs = torch.tensor(_x(x_train), dtype=torch.float32)
    ys = torch.tensor(scaler_y.transform(y_train), dtype=torch.float32)
    xp = torch.tensor(_x(x_pred), dtype=torch.float32)
    q = torch.tensor(QUANTILES, dtype=torch.float32)

    n = len(xs)
    perm = rng.permutation(n)
    n_val = max(1, round(config.val_share * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = QuantileMLP(xs.shape[1], config.hidden, config.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best, best_state, stale = float("inf"), None, 0
    for _ in range(config.max_epochs):
        model.train()
        order = rng.permutation(tr_idx)
        for i in range(0, len(order), config.batch_size):
            b = order[i : i + config.batch_size]
            opt.zero_grad()
            loss = pinball_loss(model(xs[b]), ys[b], q)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = pinball_loss(model(xs[val_idx]), ys[val_idx], q).item()
        if val < best - 1e-6:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(xp).numpy()  # (n_pred, 24, n_q), scaled space
    # inverse-transform hour by hour: the scaler is per hour column
    n_pred, _, n_q = out.shape
    flat = out.transpose(0, 2, 1).reshape(n_pred * n_q, 24)
    back = scaler_y.inverse_transform(flat).reshape(n_pred, n_q, 24).transpose(0, 2, 1)
    return back
