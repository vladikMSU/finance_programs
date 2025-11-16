from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_DATA_DIR = (Path(__file__).resolve().parent / ".." / "data").resolve()
CATEGORY_METADATA: Dict[str, Dict[str, str]] = {
    "commodities": {"asset_type": "Commodity future", "currency": "USD"},
    "crypto": {"asset_type": "Crypto asset", "currency": "USD"},
    "sp500_index": {"asset_type": "US equity ETF", "currency": "USD"},
    "us_bonds": {"asset_type": "US Treasury ETF", "currency": "USD"},
    "ru_bonds": {"asset_type": "RU bond ETF", "currency": "RUB"},
    "moex": {"asset_type": "MOEX equity", "currency": "RUB"},
    "moex_index": {"asset_type": "MOEX index", "currency": "RUB"},
}
DEFAULT_FX_FILE = DEFAULT_DATA_DIR / "currency" / "USD_RUB.csv"


# -----------------------------
# Plot style
# -----------------------------

def set_style() -> None:
    """Apply a consistent visual theme for charts."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


def styled_table(
    df: pd.DataFrame,
    *,
    caption: Optional[str] = None,
    precision: int = 3,
) -> "pd.io.formats.style.Styler":
    """Return a lightly themed styler for consistent table output in notebooks."""

    table = df.copy()

    if not table.index.is_unique:
        index_name = table.index.name or "index"
        table = table.reset_index().rename(columns={"index": index_name})

    if not table.columns.is_unique:
        counts: Dict[str, int] = {}
        unique_columns = []
        for col in table.columns:
            if col in counts:
                counts[col] += 1
                unique_columns.append(f"{col}_{counts[col]}")
            else:
                counts[col] = 0
                unique_columns.append(col)
        table.columns = unique_columns

    formatter = {col: f"{{:.{precision}f}}" for col in table.select_dtypes(include=[np.number]).columns}
    styler = table.style.format(formatter)
    if caption:
        styler = styler.set_caption(caption)

    return (
        styler.set_table_styles(
            [
                {"selector": "th", "props": "text-align: center;"},
                {"selector": "td", "props": "text-align: right;"},
                {"selector": "caption", "props": "caption-side: top; font-weight: bold;"},
            ],
            overwrite=False,
        )
        .set_properties(**{"font-size": "10pt"})
    )


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


def resample_price_frame(
    df: pd.DataFrame,
    frequency: str,
    *,
    date_col: str = "date",
    last_cols: Optional[List[str]] = None,
    sum_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate a time series to the requested frequency using last and sum logic."""
    last_cols = last_cols or []
    sum_cols = sum_cols or []

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    data["date"] = data[date_col].dt.to_period(frequency).dt.to_timestamp(how="end")

    agg: Dict[str, str] = {}
    for col in last_cols:
        agg[col] = "last"
    for col in sum_cols:
        agg[col] = "sum"

    if not agg:
        raise ValueError("At least one column must be provided in last_cols or sum_cols.")

    return data.groupby("date", as_index=False).agg(agg)


def read_price_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a simple price csv and normalise column names."""
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def build_assets_catalog(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    dir_meta: Optional[Dict[str, Dict[str, str]]] = None,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Collect asset metadata from the data directory (optionally filtered by tickers)."""
    data_dir = Path(data_dir)
    meta = dir_meta or CATEGORY_METADATA

    records: List[Dict[str, Union[str, Path]]] = []
    for category_dir in sorted(data_dir.glob("*")):
        if not category_dir.is_dir():
            continue
        if category_dir.name.startswith("."):
            continue
        if category_dir.name.lower() == "currency":
            continue
        info = meta.get(category_dir.name, {})
        asset_type_default = info.get("asset_type", category_dir.name.replace("_", " ").title())
        currency_default = info.get("currency")
        if currency_default is None:
            currency_default = "USD" if "us" in category_dir.name.lower() else "RUB"
        for csv_path in sorted(category_dir.glob("*.csv")):
            records.append(
                {
                    "asset_id": csv_path.stem,
                    "category": category_dir.name,
                    "asset_type": info.get("asset_type", asset_type_default),
                    "asset_ccy": info.get("currency", currency_default),
                    "path": csv_path,
                }
            )

    catalog = pd.DataFrame(records)
    if catalog.empty:
        raise FileNotFoundError(f"No assets discovered under {data_dir}")

    if tickers is not None:
        tickers_normalised = [t.lower() for t in tickers]
        catalog["__key"] = catalog["asset_id"].str.lower()
        filtered = catalog[catalog["__key"].isin(tickers_normalised)].copy()
        missing = sorted(set(tickers_normalised) - set(filtered["__key"]))
        if missing:
            raise FileNotFoundError(
                f"Missing data files for tickers: {', '.join(missing)}. "
                "Update CATEGORY_METADATA or ensure csv files exist."
            )
        order_map = {ticker.lower(): idx for idx, ticker in enumerate(tickers)}
        filtered["__order"] = filtered["__key"].map(order_map)
        filtered = filtered.sort_values("__order").drop(columns=["__key", "__order"])
        return filtered.reset_index(drop=True)

    return catalog.reset_index(drop=True)


def catalog_from_tickers(
    tickers: List[str],
    *,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    dir_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Convenience wrapper returning a catalog filtered to the provided tickers."""
    return build_assets_catalog(data_dir=data_dir, dir_meta=dir_meta, tickers=tickers)


def load_fx_series(
    path: Union[str, Path] = DEFAULT_FX_FILE,
    *,
    frequency: str = "Q",
) -> pd.DataFrame:
    """Load an FX series and compute log/simple returns at the requested frequency."""
    path = Path(path)
    fx = pd.read_csv(path)
    fx.columns = fx.columns.str.strip().str.lower()

    if "date" not in fx.columns:
        date_col = next((c for c in fx.columns if "date" in c), None)
        if date_col is None:
            raise ValueError(f"No date column found in {path}")
    else:
        date_col = "date"

    if "price" in fx.columns:
        price_col = "price"
    elif "close" in fx.columns:
        price_col = "close"
    else:
        raise ValueError(f"No price/close column found in {path}")

    fx = fx[[date_col, price_col]].dropna()
    fx = fx.rename(columns={date_col: "date", price_col: "fx_rate"})
    fx = resample_price_frame(fx, frequency=frequency, last_cols=["fx_rate"])
    fx["r_fx_log"] = np.log(fx["fx_rate"]).diff()
    fx["r_fx_simple"] = np.expm1(fx["r_fx_log"])
    return fx


def load_asset_series(
    asset_row: pd.Series,
    fx: pd.DataFrame,
    *,
    base_currency: str = "RUB",
    frequency: str = "Q",
) -> pd.DataFrame:
    """Load a raw asset csv, align to the requested frequency, and compute returns."""
    df = pd.read_csv(asset_row["path"])
    df.columns = df.columns.str.strip().str.lower()

    date_col = "date" if "date" in df.columns else "begin"
    if date_col not in df.columns:
        raise ValueError(f"No recognizable date column in {asset_row['path']}")

    price_col = "close" if "close" in df.columns else "price"
    if price_col not in df.columns:
        raise ValueError(f"No price/close column in {asset_row['path']}")

    keep_cols = [date_col, price_col]
    for optional in ("volume", "value"):
        if optional in df.columns:
            keep_cols.append(optional)

    df = df[keep_cols].dropna(subset=[date_col, price_col])
    df = df.rename(columns={date_col: "date", price_col: "price_local"})
    df = resample_price_frame(
        df,
        frequency=frequency,
        date_col="date",
        last_cols=["price_local"],
        sum_cols=[col for col in keep_cols if col in ("volume", "value")],
    )

    df["asset_id"] = asset_row["asset_id"]
    df["category"] = asset_row.get("category")
    df["asset_type"] = asset_row.get("asset_type")
    df["asset_ccy"] = asset_row.get("asset_ccy")

    df["r_local_log"] = np.log(df["price_local"]).diff()
    df["r_local_simple"] = np.expm1(df["r_local_log"])

    if asset_row.get("asset_ccy", "").upper() != base_currency.upper():
        fx_subset = fx[["date", "fx_rate", "r_fx_log"]]
        df = df.merge(fx_subset, on="date", how="left").sort_values("date")
        df["fx_rate"] = df["fx_rate"].ffill()
        df = df.dropna(subset=["fx_rate"])
        df["r_fx_log"] = df["r_fx_log"].fillna(0.0)
    else:
        df["fx_rate"] = 1.0
        df["r_fx_log"] = 0.0

    df["price_base"] = df["price_local"] * df["fx_rate"]
    df["r_base_log"] = df["r_local_log"] + df["r_fx_log"]
    df["r_base_simple"] = np.expm1(df["r_base_log"])
    return df


def prepare_returns_dataset(
    tickers: List[str],
    *,
    frequency: str = "Q",
    base_currency: str = "RUB",
    winsor_alpha: float = 0.01,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    fx_path: Union[str, Path] = DEFAULT_FX_FILE,
) -> Dict[str, Union[pd.DataFrame, pd.Index]]:
    """
    Load the requested tickers, align prices to `frequency`, convert to base currency,
    and return tidy/wide representations ready for analysis.
    """
    catalog = catalog_from_tickers(tickers, data_dir=data_dir)
    fx_series = load_fx_series(fx_path, frequency=frequency)

    frames: List[pd.DataFrame] = []
    for _, asset in catalog.iterrows():
        frame = load_asset_series(
            asset,
            fx_series,
            base_currency=base_currency,
            frequency=frequency,
        )
        frame["r_base_log_w"] = winsorize_series(frame["r_base_log"], winsor_alpha)
        frame["r_base_simple_w"] = np.expm1(frame["r_base_log_w"])
        frames.append(frame)

    returns_long = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["date"])
        .sort_values(["asset_id", "date"])
        .reset_index(drop=True)
    )

    counts = returns_long.groupby("date")["asset_id"].nunique()
    full_dates = counts[counts == len(tickers)].index

    aligned = returns_long[returns_long["date"].isin(full_dates)].copy()
    matrix_log = (
        aligned.pivot(index="date", columns="asset_id", values="r_base_log_w")
        .sort_index()
    )
    matrix_simple = (
        aligned.pivot(index="date", columns="asset_id", values="r_base_simple_w")
        .reindex_like(matrix_log)
    )

    return {
        "catalog": catalog,
        "fx": fx_series,
        "returns_long": returns_long,
        "aligned": aligned,
        "matrix_log": matrix_log,
        "matrix_simple": matrix_simple,
        "full_dates": matrix_log.index,
    }


def read_price_csv_monthly(path: Union[str, Path]) -> pd.DataFrame:
    """
    Back-compat helper: load a csv and aggregate to month-end closes.

    This mirrors the original read_price_csv used in earlier notebooks.
    """
    df = read_price_csv(path)
    date_col = next((c for c in df.columns if c.lower() in {"date", "timestamp", "time", "begin"}), None)
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    price_col = next(
        (c for c in ("Close", "Adj Close", "Price", "close", "AdjClose", "Last", "Close Price") if c in df.columns),
        None,
    )
    if price_col is None:
        price_col = next((c for c in df.columns if c != date_col), None)
    if price_col is None:
        raise ValueError(f"No price column found in {path}")

    prices = df[price_col]
    if prices.dtype == object:
        prices = parse_numeric(prices)

    core = df[[date_col]].copy()
    core["price"] = prices
    return resample_price_frame(core.rename(columns={date_col: "date"}), frequency="M", last_cols=["price"])


def monthly_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly log returns from a tidy price data frame."""
    price_df = price_df.sort_values("date")
    r = np.log(price_df["price"]).diff()
    return pd.DataFrame({"date": price_df["date"], "r_log": r}).dropna()


def fx_log_returns_rub_per_usd(fx_path: Union[str, Path]) -> pd.DataFrame:
    """Load USD/RUB series and compute monthly FX log and simple returns (legacy helper)."""
    fx = read_price_csv_monthly(fx_path).rename(columns={"price": "fx_rate"})
    fx["r_fx_log"] = np.log(fx["fx_rate"]).diff()
    fx["r_fx_simple"] = np.expm1(fx["r_fx_log"])
    return fx.dropna(subset=["r_fx_log"])


def fx_convert_rub(asset_ccy: str, r_local_log: pd.Series, r_fx_log: pd.Series) -> pd.Series:
    """Convert local log returns into RUB terms given USD/RUB log returns."""
    if asset_ccy.upper() == "RUB":
        return r_local_log
    return r_local_log - r_fx_log


def aggregate_quarterly_log(df: pd.DataFrame, date_col: str, value_cols: List[str]) -> pd.DataFrame:
    """Aggregate monthly log returns to calendar quarter sums."""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp["date_quarter"] = pd.PeriodIndex(tmp[date_col], freq="Q").to_timestamp(how="end")
    agg = tmp.groupby("date_quarter", as_index=False)[value_cols].sum(min_count=1)
    return agg


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
    """Weighted average for a single series aligned to the provided weights."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    w = weights.loc[valid.index]
    w = w / w.sum()
    return float(np.dot(w, valid.to_numpy()))


def weighted_covariance_matrix(data: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
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


def weighted_mean_cov(data: pd.DataFrame, weights: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
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


def ljung_box(series: pd.Series, lags: Optional[List[int]] = None) -> pd.DataFrame:
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


def plot_acf_grid(data: pd.DataFrame, max_lags: int = 12, title: str = "") -> None:
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


def plot_rolling_stats(data: pd.DataFrame, window: int = 8, title: str = "") -> None:
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

