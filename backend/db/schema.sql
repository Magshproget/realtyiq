-- ============================================================
-- RealtyIQ — Схема бази даних
-- ============================================================

-- Таблиця об'єктів нерухомості (довідник)
CREATE TABLE IF NOT EXISTS properties (
    id              SERIAL PRIMARY KEY,
    address         VARCHAR(255),
    hist_district   VARCHAR(100),
    admin_district  VARCHAR(100),
    rooms           INTEGER,
    area_total      FLOAT,
    area_living     FLOAT,
    area_kitchen    FLOAT,
    stock           INTEGER,
    stock_total     INTEGER,
    year            INTEGER,
    wall            VARCHAR(100),
    project         VARCHAR(100),
    is_complex      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Таблиця запитів на оцінку
CREATE TABLE IF NOT EXISTS estimates (
    id              SERIAL PRIMARY KEY,
    -- Вхідні параметри
    hist_district   VARCHAR(100)  NOT NULL,
    rooms           INTEGER       NOT NULL,
    area_total      FLOAT         NOT NULL,
    area_living     FLOAT,
    area_kitchen    FLOAT,
    stock           INTEGER       NOT NULL,
    stock_total     INTEGER       NOT NULL,
    year            INTEGER       NOT NULL,
    wall            VARCHAR(100),
    project         VARCHAR(100),
    is_complex      BOOLEAN       DEFAULT FALSE,
    -- Результати ML-моделі
    predicted_price_sm  FLOAT,
    predicted_total     FLOAT,
    ci_lower_95         FLOAT,
    ci_upper_95         FLOAT,
    ci_lower_80         FLOAT,
    ci_upper_80         FLOAT,
    -- Метаінформація
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Таблиця SHAP-значень для кожного запиту
CREATE TABLE IF NOT EXISTS shap_values (
    id              SERIAL PRIMARY KEY,
    estimate_id     INTEGER REFERENCES estimates(id) ON DELETE CASCADE,
    feature_name    VARCHAR(100),
    shap_value      FLOAT,
    feature_value   FLOAT
);

-- Індекси для пришвидшення запитів
CREATE INDEX IF NOT EXISTS idx_estimates_district  ON estimates(hist_district);
CREATE INDEX IF NOT EXISTS idx_estimates_created   ON estimates(created_at);
CREATE INDEX IF NOT EXISTS idx_shap_estimate_id    ON shap_values(estimate_id);
