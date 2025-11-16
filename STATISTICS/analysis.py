from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta

sns.set_theme(style="darkgrid")


@dataclass(frozen=True)
class AssetConfig:
    name: str
    price_path: Path
    currency: str = "USD"
    display_name: str | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "daily"
USD_RUB_PATH = DAILY_DATA_DIR / "currencies" / "usdrub.csv"
BASE_CURRENCY = "RUB"

ASSET_CONFIGS: Mapping[str, AssetConfig] = {
    "gold": AssetConfig(
        name="gold",
        price_path=DAILY_DATA_DIR / "commodities" / "gold.csv",
        currency="USD",
        display_name="Gold",
    ),
    "btc": AssetConfig(
        name="btc",
        price_path=DAILY_DATA_DIR / "crypto" / "btc.csv",
        currency="USD",
        display_name="Bitcoin",
    ),
    "sp500": AssetConfig(
        name="sp500",
        price_path=DAILY_DATA_DIR / "equity indices" / "US" / "sp500_TR.csv",
        currency="USD",
        display_name="S&P 500 TR",
    ),
    "MOEX": AssetConfig(
        name="MOEX",
        price_path=DAILY_DATA_DIR / "equity indices" / "RU" / "MCFTR.csv",
        currency="RUB",
        display_name="MOEX TRI",
    ),
}

TIMEFRAMES = ("D", "W-FRI", "ME", "QE", "YE")

calendar_days_per_year = 365.25
weeks_per_year = calendar_days_per_year / 7.0  # 52.1786 (calendar-based)


def read_price_history(path: Path) -> pd.DataFrame:
    """Load OHLC price history and ensure Date is the index."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    return df


def load_usd_rub_series() -> pd.Series:
    fx_df = read_price_history(USD_RUB_PATH)
    fx = fx_df["Close"].rename("USD/RUB").sort_index()
    return fx


def convert_to_base_currency(
    df: pd.DataFrame,
    asset_currency: str,
    fx_rates: Mapping[str, pd.Series],
    base_currency: str = BASE_CURRENCY,
) -> pd.DataFrame:
    """Convert prices to the base currency (currently RUB)."""
    if asset_currency == base_currency:
        df.attrs["currency"] = base_currency
        return df

    if asset_currency == "USD" and base_currency == "RUB":
        fx_series = fx_rates["USD/RUB"]
        aligned_fx = fx_series.reindex(df.index).ffill().bfill()
        df = df.copy()
        df["Close_USD"] = df["Close"]
        df["Close"] = df["Close_USD"] * aligned_fx
        df.attrs["currency"] = base_currency
        return df

    raise NotImplementedError(
        f"Conversion from {asset_currency} to {base_currency} is not implemented."
    )


def resample_price_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    resampled = df.resample(freq).last().dropna(subset=["Close"]).copy()
    resampled["returns"] = resampled["Close"].pct_change()
    resampled["log_returns"] = np.log(resampled["Close"]).diff()
    resampled = resampled.dropna(subset=["returns"])
    return resampled


def prepare_asset_frames(
    configs: Mapping[str, AssetConfig],
    timeframes: Iterable[str],
    fx_rates: Mapping[str, pd.Series],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    prepared: Dict[str, Dict[str, pd.DataFrame]] = {}
    for asset_name, config in configs.items():
        price_df = read_price_history(config.price_path)
        price_df = convert_to_base_currency(price_df, config.currency, fx_rates)
        prepared[asset_name] = {
            freq: resample_price_series(price_df, freq) for freq in timeframes
        }
    return prepared


def calc_geom_annual_return(df: pd.DataFrame) -> float:
    s = df["Close"]
    start_price, end_price = float(s.iloc[0]), float(s.iloc[-1])
    total_factor = end_price / start_price
    start_date, end_date = s.index[0], s.index[-1]
    rd = relativedelta(end_date, start_date)
    years_float = rd.years + rd.months / 12 + rd.days / calendar_days_per_year
    return total_factor ** (1 / years_float) - 1.0


def calc_simple_annual_return(df: pd.DataFrame) -> float:
    s = df["Close"]
    start_price, end_price = float(s.iloc[0]), float(s.iloc[-1])
    total_return = (end_price - start_price) / start_price
    start_date, end_date = s.index[0], s.index[-1]
    rd = relativedelta(end_date, start_date)
    years_float = rd.years + rd.months / 12 + rd.days / calendar_days_per_year
    return total_return / years_float


def calc_annual_stddev(df: pd.DataFrame) -> tuple[float, float]:
    s = df["Close"]
    anchor = "W-FRI"  # use Friday as week anchor
    weekly_prices = s.resample(anchor).last()
    weekly_returns = weekly_prices.pct_change().dropna()
    weekly_stddev = weekly_returns.std()
    annualized_stddev_simple = weekly_stddev * (weeks_per_year**0.5)

    weekly_log_returns = np.log(weekly_prices / weekly_prices.shift(1)).dropna()
    weekly_log_stddev = weekly_log_returns.std()
    annualized_stddev_log = weekly_log_stddev * (weeks_per_year**0.5)
    return annualized_stddev_simple, annualized_stddev_log


def summarize_returns(df: pd.DataFrame) -> pd.Series:
    """Convenience statistics for a single resampled dataframe."""
    simple_mean = df["returns"].mean()
    compounded = (1 + df["returns"]).prod() ** (1 / len(df)) - 1
    std = df["returns"].std()
    skew = df["returns"].skew()
    kurt = df["returns"].kurtosis()
    return pd.Series(
        {
            "simple_mean": simple_mean,
            "cagr_from_returns": compounded,
            "std": std,
            "skew": skew,
            "kurtosis": kurt,
        }
    )


def plot_price_history(df: pd.DataFrame, title: str | None = None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    df["Close"].plot(ax=ax, title=title or "Price history")
    ax.set_ylabel(f"Close ({df.attrs.get('currency', 'N/A')})")
    return ax


def plot_return_distribution(
    df: pd.DataFrame, column: str = "returns", bins: int = 50, title: str | None = None
):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], bins=bins, ax=ax, kde=True)
    ax.set_title(title or f"{column} distribution")
    return ax


def main():
    fx_rates = {"USD/RUB": load_usd_rub_series()}
    asset_frames = prepare_asset_frames(ASSET_CONFIGS, TIMEFRAMES, fx_rates)

    for asset_name, freq_dict in asset_frames.items():
        print("-" * 60)
        print(asset_name.upper())
        for freq, df in freq_dict.items():
            stats = summarize_returns(df)
            print(
                f"{freq:>5} | {len(df):5} rows | "
                f"mean: {stats['simple_mean']:.4%} | "
                f"CAGR: {stats['cagr_from_returns']:.4%} | "
                f"std: {stats['std']:.4%}"
            )

    # Quick sanity plots for iterative analysis.
    example_asset = asset_frames["sp500"]["ME"]
    plot_price_history(example_asset, title="S&P 500 (monthly close, RUB)")
    plot_return_distribution(example_asset, title="S&P 500 monthly returns")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
