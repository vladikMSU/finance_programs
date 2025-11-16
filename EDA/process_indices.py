import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

indices_folder_path = os.path.join("..", "data", "indices")

ru_idices_meta = pd.read_csv(os.path.join(indices_folder_path, "indices_meta.csv"))

indices = {}
def parse_indices_meta():
    indices_meta = pd.read_csv(os.path.join(indices_folder_path, "indices_meta.csv"))
    for idx, row in indices_meta.iterrows():
        ticker = row['Тикер']
        description = row['Описание']
        indices[description] = ticker
    return indices_meta

def reshape_and_calc_returns(df, price_col='Close', resample_factor='W-FRI'):
    s = df[price_col]
    df_reshaped = pd.DataFrame()

    df_reshaped['Price'] = s.resample(resample_factor).last()
    df_reshaped['Return'] = df_reshaped['Price'].pct_change()
    df_reshaped['Log Return'] = np.log(df_reshaped['Prices']).diff()

    return df_reshaped

def calc_rolling_volatility(df, return_col='Return', window=4):
    df_vol = pd.DataFrame()
    df_vol['Rolling Volatility'] = df[return_col].rolling(window=window).std() * np.sqrt(window)
    return df_vol

def calc_geom_annual_return(df: pd.DataFrame) -> pd.DataFrame:
    s = df['Close']

    start_price, end_price = float(s.iloc[0]), float(s.iloc[-1])
    total_factor = end_price / start_price

    start_date, end_date = s.index[0], s.index[-1]
    rd = relativedelta(end_date, start_date)
    years_float = rd.years + rd.months/12 + rd.days/calendar_days_per_year   # fractional approx
    
    return total_factor ** (1 / years_float) - 1.0

def calc_simple_annual_return(df: pd.DataFrame) -> pd.DataFrame:
    s = df['Close']

    start_price, end_price = float(s.iloc[0]), float(s.iloc[-1])
    total_return = (end_price - start_price) / start_price

    start_date, end_date = s.index[0], s.index[-1]
    rd = relativedelta(end_date, start_date)
    years_float = rd.years + rd.months/12 + rd.days/calendar_days_per_year   # fractional approx
    
    return total_return / years_float

def calc_annual_stddev(df: pd.DataFrame) -> pd.DataFrame:
    s = df['Close']
    anchor = 'W-FRI'  # use Friday as week anchor
    weekly_prices = s.resample(anchor).last()

    weekly_returns = weekly_prices.pct_change().dropna()
    weekly_stddev = weekly_returns.std()
    annualized_stddev_simple = weekly_stddev * (weeks_per_year ** 0.5)

    weekly_log_returns = (weekly_prices / weekly_prices.shift(1)).apply(lambda x: pd.np.log(x)).dropna()
    weekly_log_stddev = weekly_log_returns.std()
    annualized_stddev_log = weekly_log_stddev * (weeks_per_year ** 0.5)
    # daily_returns = s.pct_change().dropna()
    return annualized_stddev_simple, annualized_stddev_log
    

def read_and_process_indices(indices_meta):
    bm_index_df = pd.read_csv(os.path.join(base_dir, f"{indices_meta.iloc[0]["Тикер"]}.csv"), parse_dates=["Date"], index_col="Date")  

    base_main_index = pd.read_csv(os.path.join(base_dir, f"{indices_meta.iloc[1]["Тикер"]}.csv"), parse_dates=["Date"], index_col="Date")  
    tr_main_index = pd.read_csv(os.path.join(base_dir, f"{indices_meta.iloc[1]["TR брутто"]}.csv"), parse_dates=["Date"], index_col="Date")  

    sm_index = pd.read_csv(os.path.join(base_dir, f"{indices_meta.iloc[2]["Тикер"]}.csv"), parse_dates=["Date"], index_col="Date")
    tr_sm_index = pd.read_csv(os.path.join(base_dir, f"{indices_meta.iloc[2]["TR брутто"]}.csv"), parse_dates=["Date"], index_col="Date")
    
    base_industry_indices_dfs = {}
    tr_industry_indices_dfs = {}
    for ind, row in ru_idices_meta.iloc[3:].iterrows():
        industry = row["Описание"]
        
        base_index = row['Тикер']
        tr_index = row['TR брутто']
        
        base_index_path = os.path.join(base_dir, f"{base_index}.csv")
        base_industry_indices_dfs[industry] = pd.read_csv(base_index_path, parse_dates=["Date"], index_col="Date")

        tr_index_path = os.path.join(base_dir, f"{tr_index}.csv")
        tr_industry_indices_dfs[industry] = pd.read_csv(tr_index_path, parse_dates=["Date"], index_col="Date")

calendar_days_per_year = 365.25
weeks_per_year = calendar_days_per_year / 7.0  # ≈ 52.1786 (calendar-based)


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== ПУТИ И ЧТЕНИЕ МЕТА =====
indices_folder_path = os.path.join("..", "data", "indices")
meta = pd.read_csv(os.path.join(indices_folder_path, "indices_meta.csv"))  # ожидаем колонки: "Тикер", "TR брутто"

meta["TR брутто"] = meta.apply(lambda row: np.nan if row["TR брутто"] == "нет" else row["TR брутто"], axis=1)

# ===== МИНИ-РИДЕР =====
def read_series(indice_ticker: str) -> pd.Series:
    df = pd.read_csv(os.path.join(indices_folder_path, f'{indice_ticker}.csv'))
    df["Date"] = pd.to_datetime(df["Date"])
    s = (df.set_index("Date")["Close"].astype(float).sort_index())
    s.name = indice_ticker
    return s

# ===== ЗАГРУЗКА PX/TR В ШИРОКИЕ МАТРИЦЫ =====
px_dict, tr_dict = {}, {}
for _, row in meta.iterrows():
    t = row["Тикер"]
    px_dict[t] = read_series(t)

    tr_name = row.get("TR брутто", np.nan)
    if pd.notna(tr_name):
        tr_dict[t] = read_series(tr_name)

def to_wide(d: dict) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    w = pd.concat(d, axis=1)
    w.index.name, w.columns.name = "Date", "Ticker"
    return w

px_wide = to_wide(px_dict)     # цены обычных индексов
tr_wide = to_wide(tr_dict)     # цены TR-пар
# Главная матрица цен: TR если есть, иначе PX
tickers = sorted(set(px_dict) | set(tr_dict))
pref_wide = to_wide({t: tr_dict.get(t, px_dict.get(t)) for t in tickers})

# ===== ДОХОДНОСТИ И МЕТРИКИ =====
def returns(df: pd.DataFrame, log: bool = True, resample: str | None = None) -> pd.DataFrame:
    """Лог- или простые доходности; опционально ресемплим цены и берём last."""
    x = df.copy()
    if resample:
        x = x.resample(resample).last()
    r = np.log(x).diff() if log else x.pct_change()
    return r.dropna(how="all")

def annualize(mu_per, vol_per, ppy: int) -> tuple[pd.Series, pd.Series]:
    """Превращаем среднюю доходность и волу per-period → annualized."""
    mu_ann  = mu_per  * ppy
    vol_ann = vol_per * np.sqrt(ppy)
    return mu_ann, vol_ann

# Пример: недельные лог-доходности и базовые метрики
R_week = returns(pref_wide, log=True, resample="W-FRI")
mu_w, vol_w = R_week.mean(), R_week.std(ddof=1)
mu_ann, vol_ann = annualize(mu_w, vol_w, ppy=52)
corr = R_week.corr()

# ===== БЫСТРЫЕ ГРАФИКИ ДЛЯ СТАТЬИ/EDA =====
def plot_normalized(df: pd.DataFrame, tickers: list[str] | None = None, title="Normalized (base=100)", save: str | None = None):
    sel = df[tickers] if tickers else df
    norm = sel / sel.iloc[0] * 100
    ax = norm.plot(figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Index, base=100")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    plt.show()

def plot_cum_from_log_ret(R: pd.DataFrame, title="Cumulative (from log-returns)", save: str | None = None):
    cum = np.exp(R.cumsum())
    ax = cum.plot(figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Growth factor")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    plt.show()

# ==== ПРИМЕР ИСПОЛЬЗОВАНИЯ (можно комментировать/раскомментировать) ====
# 1) Нормированные цены (предпочтительно TR, если есть)
# plot_normalized(pref_wide, title="All indices (preferred TR) normalized")

# 2) Кумулятивы из лог-доходностей (weekly)
# plot_cum_from_log_ret(R_week, title="Cumulative perf (weekly log-returns)")

# 3) Топ-10 по среднегодовой доходности/волатильности
# print(mu_ann.sort_values(ascending=False).round(3).head(10))
# print(vol_ann.sort_values(ascending=False).round(3).head(10))
# print(corr.round(2))