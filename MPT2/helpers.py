from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro


# -----------------------------
# Plot style
# -----------------------------

def set_style() -> None:
    """Apply a consistent visual theme for charts."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


# -----------------------------
# Data utilities
# -----------------------------

def parse_numeric(series: pd.Series) -> pd.Series:
    """Convert messy numeric strings (spaces, commas, unicode minus) into floats."""
    cleaned = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("\u2212", "-", regex=False)
        .str.replace("?", "-", regex=False)
        .replace({"": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_price_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a price csv, tidy the columns, and aggregate to month-end closes."""
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    date_col = None
    for col in df.columns:
        if col.lower() in {"date", "timestamp", "time", "begin"}:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)

    price_col = None
    for candidate in (
        "Close",
        "Adj Close",
        "Price",
        "close",
        "AdjClose",
        "Last",
        "Close Price",
    ):
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        for col in df.columns:
            if col not in {date_col, "date"}:
                price_col = col
                break
    if price_col is None:
        raise ValueError(f"No price column found in {path}")

    prices = df[price_col]
    if prices.dtype == object:
        prices = parse_numeric(prices)

    out = (
        pd.DataFrame({"date": df["date"], "price": prices})
        .dropna()
        .sort_values("date")
    )
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp(how="end")
    out = out.groupby("date", as_index=False).last()
    out = out[out["price"] > 0]
    return out


def monthly_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly log returns from a tidy price data frame."""
    price_df = price_df.sort_values("date")
    r = np.log(price_df["price"]).diff()
    return pd.DataFrame({"date": price_df["date"], "r_log": r}).dropna()


def fx_log_returns_rub_per_usd(fx_path: Union[str, Path]) -> pd.DataFrame:
    """Load USD/RUB series and compute monthly FX log and simple returns."""
    fx = read_price_csv(fx_path).rename(columns={"price": "fx_rate"})
    fx["r_fx_log"] = np.log(fx["fx_rate"]).diff()
    fx["r_fx_simple"] = np.expm1(fx["r_fx_log"])
    return fx.dropna(subset=["r_fx_log"])


def fx_convert_rub(asset_ccy: str,
                   r_local_log: pd.Series,
                   r_fx_log: pd.Series) -> pd.Series:
    """Convert local log returns into RUB terms given USD/RUB log returns."""
    if asset_ccy.upper() == "RUB":
        return r_local_log
    # USD quoted vs RUB: base log return = local log return - FX log return
    return r_local_log - r_fx_log


def aggregate_quarterly_log(df: pd.DataFrame,
                            date_col: str,
                            value_cols: List[str]) -> pd.DataFrame:
    """Aggregate monthly log returns to calendar quarter sums."""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp["date_quarter"] = pd.PeriodIndex(tmp[date_col], freq="Q").to_timestamp(how="end")
    agg = tmp.groupby("date_quarter", as_index=False)[value_cols].sum(min_count=1)
    return agg


def build_assets_catalog(data_dir: Union[str, Path],
                         dir_meta: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Create a metadata table for all csv assets in the data directory."""
    data_dir = Path(data_dir)
    entries = []
    for category, meta in dir_meta.items():
        for csv_path in sorted((data_dir / category).glob("*.csv")):
            entries.append({
                "asset_id": csv_path.stem,
                "category": category,
                "asset_type": meta.get("asset_type"),
                "asset_ccy": meta.get("currency"),
                "path": csv_path,
            })
    return pd.DataFrame(entries)


def load_fx_series(path: Union[str, Path],
                   frequency: str = "M") -> pd.DataFrame:
    """Load an FX series and compute log/simple returns at the chosen frequency."""
    fx = pd.read_csv(path)
    fx.columns = fx.columns.str.strip().str.lower()
    if "date" not in fx.columns:
        raise ValueError("FX file must contain a date column.")
    price_col = "price" if "price" in fx.columns else "close"
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    fx["fx_rate"] = parse_numeric(fx[price_col])
    fx = fx[["date", "fx_rate"]].dropna().sort_values("date")
    fx["date"] = fx["date"].dt.to_period(frequency).dt.to_timestamp(how="end")
    fx = fx.groupby("date", as_index=False).last()
    fx["r_fx_log"] = np.log(fx["fx_rate"]).diff()
    fx["r_fx_simple"] = np.expm1(fx["r_fx_log"])
    return fx.dropna(subset=["r_fx_log"])


def load_asset_series(asset_row: pd.Series,
                      fx: pd.DataFrame,
                      base_currency: str = "RUB",
                      frequency: str = "M") -> pd.DataFrame:
    """Load a raw asset csv, align to month-end, and compute RUB returns."""
    df = pd.read_csv(asset_row["path"])
    df.columns = df.columns.str.strip().str.lower()

    date_col = "date" if "date" in df.columns else "begin"
    if date_col not in df.columns:
        raise ValueError(f"No recognizable date column in {asset_row['path']}")
    price_col = "close" if "close" in df.columns else "price"

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    keep_cols = ["date", price_col]
    for optional in ("volume", "value"):
        if optional in df.columns:
            keep_cols.append(optional)
    df = df[keep_cols].dropna(subset=["date", price_col]).sort_values("date")
    df["date"] = df["date"].dt.to_period(frequency).dt.to_timestamp(how="end")

    agg = {price_col: "last"}
    if "volume" in keep_cols:
        agg["volume"] = "sum"
    df = df.groupby("date", as_index=False).agg(agg)
    df = df.rename(columns={price_col: "price_local"})

    df["asset_id"] = asset_row["asset_id"]
    df["category"] = asset_row.get("category")
    df["asset_type"] = asset_row.get("asset_type")
    df["asset_ccy"] = asset_row.get("asset_ccy")

    if asset_row.get("asset_ccy", "").upper() != base_currency.upper():
        fx_subset = fx[["date", "fx_rate", "r_fx_log"]]
        df = df.merge(fx_subset, on="date", how="left").sort_values("date")
        df["fx_rate"] = df["fx_rate"].ffill()
        df = df.dropna(subset=["fx_rate"])
        df["r_fx_log"] = df["r_fx_log"].fillna(0.0)
        df["price_base"] = df["price_local"] * df["fx_rate"]
    else:
        df["fx_rate"] = 1.0
        df["r_fx_log"] = 0.0
        df["price_base"] = df["price_local"]

    df["r_local_log"] = np.log(df["price_local"]).diff()
    df["r_base_log"] = df["r_local_log"] + df["r_fx_log"]
    df["r_local_simple"] = np.expm1(df["r_local_log"])
    df["r_base_simple"] = np.expm1(df["r_base_log"])
    return df


# -----------------------------
# Stats utilities
# -----------------------------

def winsorize_series(series: pd.Series, alpha: float) -> pd.Series:
    """Clip both tails of a series at the specified alpha level."""
    values = series.dropna()
    if values.empty:
        return series
    lower = values.quantile(alpha)
    upper = values.quantile(1 - alpha)
    return series.clip(lower, upper)


def exp_weights(index: pd.Index, half_life: float) -> pd.Series:
    """Exponentially decaying weights (newest observation has most weight)."""
    if len(index) == 0:
        return pd.Series(dtype=float)
    lam = 2 ** (-1.0 / half_life)
    exponents = np.arange(len(index))[::-1]
    weights = lam ** exponents
    weights = weights / weights.sum()
    return pd.Series(weights, index=index)


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Weighted average for a single series aligned to weights."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    w = weights.loc[valid.index]
    w = w / w.sum()
    return float(np.dot(w, valid.to_numpy()))


def weighted_covariance_matrix(data: pd.DataFrame,
                               weights: pd.Series) -> pd.DataFrame:
    """Pairwise exponentially weighted covariance matrix."""
    cols = data.columns
    cov = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i, col_i in enumerate(cols):
        series_i = data[col_i]
        for j in range(i, len(cols)):
            col_j = cols[j]
            series_j = data[col_j]
            pair = pd.concat([series_i, series_j], axis=1).dropna()
            if len(pair) < 2:
                continue
            w = weights.loc[pair.index]
            w = w / w.sum()
            x = pair.iloc[:, 0].to_numpy()
            y = pair.iloc[:, 1].to_numpy()
            mx = np.dot(w, x)
            my = np.dot(w, y)
            denom = 1 - np.sum(np.square(w))
            if denom <= 0:
                continue
            cov_ij = np.dot(w, (x - mx) * (y - my)) / denom
            cov.iloc[i, j] = cov_ij
            cov.iloc[j, i] = cov_ij
    return cov


def weighted_mean_cov(data: pd.DataFrame,
                      weights: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """Convenience wrapper returning both weighted mean vector and covariance."""
    aligned_weights = weights.loc[data.index]
    aligned_weights = aligned_weights / aligned_weights.sum()
    mu = pd.Series(
        {col: weighted_mean(data[col], aligned_weights) for col in data.columns},
        name="mu",
    )
    cov = weighted_covariance_matrix(data, aligned_weights)
    return mu, cov


def nearest_psd(matrix: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Project a covariance estimate onto the PSD cone."""
    array = matrix.fillna(0.0).to_numpy()
    array = (array + array.T) / 2
    eigvals, eigvecs = np.linalg.eigh(array)
    eigvals = np.maximum(eigvals, eps)
    rebuilt = (eigvecs * eigvals) @ eigvecs.T
    return pd.DataFrame(rebuilt, index=matrix.index, columns=matrix.columns)


def jb_test(series: pd.Series) -> Tuple[float, float]:
    """Jarque-Bera test statistic and p-value."""
    stat, p, _, _ = jarque_bera(series.dropna())
    return float(stat), float(p)


def shapiro_test(series: pd.Series) -> Tuple[float, float]:
    """Shapiro-Wilk test statistic and p-value."""
    try:
        stat, p = shapiro(series.dropna())
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def ljung_box(series: pd.Series,
              lags: List[int] = None) -> pd.DataFrame:
    """Ljung-Box test across the requested lags."""
    if lags is None:
        lags = [4, 8]
    return acorr_ljungbox(series.dropna(), lags=lags, return_df=True)


def compute_ledoit_wolf(data: pd.DataFrame):
    """Ledoit-Wolf shrinkage covariance on the full intersection of data."""
    from sklearn.covariance import LedoitWolf

    clean = data.dropna()
    if len(clean) < 2 or clean.shape[0] <= clean.shape[1]:
        return None
    lw = LedoitWolf().fit(clean.to_numpy())
    return pd.DataFrame(lw.covariance_, index=data.columns, columns=data.columns)


# -----------------------------
# Plot utilities
# -----------------------------

def plot_correlation_heatmap(Sigma_psd: pd.DataFrame, title: str = "") -> None:
    corr = Sigma_psd.corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(title or "Correlation heatmap")
    plt.tight_layout()
    plt.show()


def plot_acf_grid(data: pd.DataFrame,
                  max_lags: int = 12,
                  title: str = "") -> None:
    n_assets = data.shape[1]
    columns = list(data.columns)
    ncols = 3
    nrows = int(np.ceil(n_assets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        series = data[col].dropna()
        if len(series) > 2:
            plot_acf(series, lags=min(max_lags, len(series) - 2), ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title or "ACF per asset", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_rolling_stats(data: pd.DataFrame,
                       window: int = 8,
                       title: str = "") -> None:
    plt.figure(figsize=(9, 4))
    for col in data.columns:
        series = data[col].dropna()
        if len(series) >= window:
            plt.plot(series.index, series.rolling(window).std(), label=col)
    plt.title(title or f"Rolling stdev (window={window})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_drawdowns(data: pd.DataFrame, title: str = "") -> None:
    plt.figure(figsize=(9, 4))
    for col in data.columns:
        series = data[col].dropna()
        if series.empty:
            continue
        log_price = series.cumsum()
        peak = log_price.cummax()
        drawdown = log_price - peak
        plt.plot(drawdown.index, drawdown.values, label=col)
    plt.title(title or "Drawdowns (log)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def split_crypto_noncrypto(columns: List[str]) -> Tuple[List[str], List[str]]:
    """Simple heuristic to split crypto tickers (ending with -USD)."""
    crypto = [col for col in columns if col.upper().endswith("-USD")]
    noncrypto = [col for col in columns if col not in crypto]
    return crypto, noncrypto

