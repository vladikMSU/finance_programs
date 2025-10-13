#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import cvxpy as cp


# # Шаг 1: Загружаем данные

# In[149]:


tickers = ['AAPL', 'MSFT', 'AMZN',] #'TSLA']
data = yf.download(tickers, start="1995-01-01", end="2025-07-02", interval='1mo')


# In[150]:


data.head(3)


# In[151]:


data.tail(3)


# In[152]:


def resample_open_to_close_returns(df: pd.DataFrame, freq='YE') -> pd.DataFrame:
    """
    Считает доходности между Open начала периода и Close конца периода.
    Ожидается DataFrame с колонками MultiIndex: [('AAPL', 'Open'), ('AAPL', 'Close'), ...]
    """
    # Убедимся, что формат — MultiIndex: тикеры × ['Open', 'Close']
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    tickers = df.columns.get_level_values(1).unique()
    result = pd.DataFrame(index=[], columns=tickers)

    for ticker in tickers:
        open_series = df['Open'][ticker].resample(freq).first()
        close_series = df['Close'][ticker].resample(freq).last()
        ret = (close_series / open_series - 1).dropna()
        result[ticker] = ret

    return result


# In[153]:


# Доходности Open → Close на год
freq = 'YE' # 'QE'
returns = resample_open_to_close_returns(data, freq='YE')


# In[154]:


returns


# # 2. Вычисление статистик

# In[155]:


periods_back = 7 #len(returns)
count_2025 = -1 #len(returns)
returns.iloc[-periods_back:count_2025]


# In[156]:


mean_returns = returns.iloc[-periods_back:count_2025].mean()
returns_std = returns.iloc[-periods_back:count_2025].std()
cov_matrix = returns.iloc[-periods_back:count_2025].cov()


# In[157]:


mean_returns


# In[158]:


returns_std


# In[159]:


cov_matrix


# In[160]:


returns.corr()


# # 3. Симуляция случайных портфелей

# In[161]:


n_portfolios = 50000
results = np.zeros((3, n_portfolios))


# In[162]:


get_ipython().run_cell_magic('time', '', 'for i in range(n_portfolios):\n    weights = np.random.rand(len(tickers))\n    weights /= np.sum(weights)\n\n    port_return = np.dot(weights, mean_returns)\n    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))\n    sharpe = port_return / port_std\n\n    results[0, i] = port_return\n    results[1, i] = port_std\n    results[2, i] = sharpe\n')


# # 4. Визуализация

# In[163]:


plt.scatter(results[1], results[0], c=results[2], cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier (Random Portfolios)')
plt.grid(True)
plt.show()


# # 5. Оптимальный портфель при заданной доходности (библиотека cvxpy)

# In[164]:


def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, std


# In[165]:


def optimizer_function(cov_matrix, ret_target):
    n = len(cov_matrix)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1,
                   mean_returns.values @ w >= ret_target,
                   w >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w


# In[166]:


w = optimizer_function(cov_matrix, 0.40)

# === 8. Вывод весов оптимального портфеля ===
print("Optimal weights (min volatility):")
for ticker, weight in zip(tickers, w.value):
    print(f"{ticker}: {weight:.2%}")


# In[167]:


opt_return, opt_std = portfolio_performance(w.value, mean_returns, cov_matrix)


# In[168]:


opt_return, opt_std


# In[169]:


returns_std


# In[170]:


# === 6. Визуализация ===
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.5)
plt.scatter(opt_std, opt_return, color='red', marker='*', s=200, label='min volatility')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




