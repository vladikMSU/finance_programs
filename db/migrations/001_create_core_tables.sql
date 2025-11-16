-- 001_create_core_tables.sql
-- Defines foundational dimension and fact tables for price tracking.

-- Ensure the custom enum exists so asset types remain validated.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'asset_type_enum') THEN
        CREATE TYPE asset_type_enum AS ENUM ('gold', 'bitcoin', 'equity', 'bond', 'fx');
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS countries (
    id              BIGSERIAL PRIMARY KEY,
    iso_code        CHAR(2)    NOT NULL UNIQUE,
    name            TEXT       NOT NULL,
    currency_code   CHAR(3)    NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (iso_code = UPPER(iso_code)),
    CHECK (currency_code = UPPER(currency_code))
);

CREATE TABLE IF NOT EXISTS assets (
    id                   BIGSERIAL PRIMARY KEY,
    asset_code           TEXT           NOT NULL UNIQUE,
    display_name         TEXT           NOT NULL,
    asset_type           asset_type_enum NOT NULL,
    base_currency_code   CHAR(3)        NOT NULL,
    country_id           BIGINT         REFERENCES countries(id),
    bloomberg_ticker     TEXT,
    metadata             JSONB          DEFAULT '{}'::JSONB,
    created_at           TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    CHECK (base_currency_code = UPPER(base_currency_code)),
    CHECK (char_length(asset_code) > 0)
);

CREATE TABLE IF NOT EXISTS asset_daily_prices (
    id              BIGSERIAL PRIMARY KEY,
    asset_id        BIGINT      NOT NULL REFERENCES assets(id) ON UPDATE CASCADE ON DELETE CASCADE,
    price_date      DATE        NOT NULL,
    close_price     NUMERIC(18,6) NOT NULL,
    currency_code   CHAR(3)     NOT NULL,
    data_vendor     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (close_price >= 0),
    CHECK (currency_code = UPPER(currency_code)),
    CHECK (price_date >= DATE '1900-01-01')
);

-- Uniqueness & lookup helpers
CREATE UNIQUE INDEX IF NOT EXISTS idx_asset_daily_prices_unique
    ON asset_daily_prices (asset_id, price_date);

CREATE INDEX IF NOT EXISTS idx_assets_type_country
    ON assets (asset_type, country_id);

CREATE INDEX IF NOT EXISTS idx_asset_daily_prices_asset_date
    ON asset_daily_prices (asset_id, price_date);

COMMENT ON TABLE countries IS 'Reference list of sovereign countries used for tagging assets.';
COMMENT ON TABLE assets IS 'Tradable instruments tracked across the finance programs workspace.';
COMMENT ON TABLE asset_daily_prices IS 'Daily closes (or last available price) for supported assets.';
