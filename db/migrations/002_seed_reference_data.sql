-- 002_seed_reference_data.sql
-- Inserts baseline countries and assets so analysts can demo queries immediately.

INSERT INTO countries (iso_code, name, currency_code)
VALUES
    ('US', 'United States', 'USD'),
    ('GB', 'United Kingdom', 'GBP'),
    ('JP', 'Japan', 'JPY'),
    ('RU', 'Russia', 'RUB'),
    ('CN', 'China', 'CNY'),
    ('AE', 'United Arab Emirates', 'AED')
ON CONFLICT (iso_code) DO UPDATE
SET name = EXCLUDED.name,
    currency_code = EXCLUDED.currency_code,
    updated_at = NOW();

WITH country_lookup AS (
    SELECT iso_code, id FROM countries WHERE iso_code IN ('US','GB','JP','RU','CN','AE')
)
INSERT INTO assets (
    asset_code,
    display_name,
    asset_type,
    base_currency_code,
    country_id,
    bloomberg_ticker
)
VALUES
    ('XAUUSD', 'LBMA Gold Price (USD)', 'gold', 'USD', (SELECT id FROM country_lookup WHERE iso_code = 'US'), 'XAUUSD:CUR'),
    ('BTCUSD', 'Bitcoin / USD', 'bitcoin', 'USD', NULL, 'BTCUSD:CUR'),
    ('SP500', 'S&P 500 Index', 'equity', 'USD', (SELECT id FROM country_lookup WHERE iso_code = 'US'), 'SPX:IND'),
    ('IMOEX', 'MOEX Russia Index', 'equity', 'RUB', (SELECT id FROM country_lookup WHERE iso_code = 'RU'), 'IMOEX:IND'),
    ('US10Y', 'US 10Y Treasury', 'bond', 'USD', (SELECT id FROM country_lookup WHERE iso_code = 'US'), 'USGG10YR:IND'),
    ('EURUSD', 'EUR / USD FX', 'fx', 'USD', (SELECT id FROM country_lookup WHERE iso_code = 'GB'), 'EURUSD:CUR')
ON CONFLICT (asset_code) DO UPDATE
SET display_name = EXCLUDED.display_name,
    asset_type = EXCLUDED.asset_type,
    base_currency_code = EXCLUDED.base_currency_code,
    country_id = EXCLUDED.country_id,
    bloomberg_ticker = EXCLUDED.bloomberg_ticker,
    updated_at = NOW();
