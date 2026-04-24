# -*- coding: utf-8 -*-
"""
predict.py — FastAPI ML-сервiс
Приймає параметри квартири, повертає прогноз + SHAP + iнтервали
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
import uvicorn

# -- Шляхи -------------------------------------------------------------------
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE, 'ml', 'model', 'xgboost_model.pkl')
INTERVAL_PATH = os.path.join(BASE, 'ml', 'model', 'interval_results.json')

# -- Завантаження моделі -----------------------------------------------------
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

with open(INTERVAL_PATH, 'r') as f:
    interval_meta = json.load(f)

preprocessor = pipeline.named_steps['pre']
model        = pipeline.named_steps['model']
explainer    = shap.TreeExplainer(model)

NUM_FEATURES = [
    'area_total', 'area_living', 'area_kitchen',
    'rooms', 'stock', 'stock_total', 'year', 'building_age',
    'floor_ratio', 'is_new_building', 'is_complex',
    'room_area', 'kitchen_ratio', 'living_ratio',
    'is_first_floor', 'is_last_floor'
]
CAT_FEATURES = ['hist_district', 'wall', 'project']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

NAME_MAP = {
    'area_total':      'Загальна площа',
    'area_living':     'Житлова площа',
    'area_kitchen':    'Площа кухні',
    'rooms':           'К-сть кімнат',
    'stock':           'Поверх',
    'stock_total':     'Поверховість будинку',
    'year':            'Рік побудови',
    'building_age':    'Вік будинку',
    'floor_ratio':     'Відносний поверх',
    'is_new_building': 'Нова забудова',
    'is_complex':      'Належить до ЖК',
    'room_area':       'Площа на кімнату',
    'kitchen_ratio':   'Частка кухні',
    'living_ratio':    'Частка житлової площі',
    'is_first_floor':  'Перший поверх',
    'is_last_floor':   'Останній поверх',
    'hist_district':   'Історичний район',
    'wall':            'Матеріал стін',
    'project':         'Тип проекту',
}

# -- FastAPI -----------------------------------------------------------------
app = FastAPI(title='RealtyIQ ML Service', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)

class PropertyInput(BaseModel):
    hist_district: str
    rooms:         int
    area_total:    float
    area_living:   float  = None
    area_kitchen:  float  = None
    stock:         int
    stock_total:   int
    year:          int
    wall:          str    = 'Невідомо'
    project:       str    = 'Невідомо'
    is_complex:    int    = 0

def build_features(p: PropertyInput) -> pd.DataFrame:
    area_living  = p.area_living  or p.area_total * 0.6
    area_kitchen = p.area_kitchen or 10.0

    row = {
        'area_total':      p.area_total,
        'area_living':     area_living,
        'area_kitchen':    area_kitchen,
        'rooms':           p.rooms,
        'stock':           p.stock,
        'stock_total':     p.stock_total,
        'year':            p.year,
        'building_age':    2024 - p.year,
        'floor_ratio':     p.stock / max(p.stock_total, 1),
        'is_new_building': int(p.year >= 2000),
        'is_complex':      p.is_complex,
        'room_area':       p.area_total / max(p.rooms, 1),
        'kitchen_ratio':   area_kitchen / p.area_total,
        'living_ratio':    area_living  / p.area_total,
        'is_first_floor':  int(p.stock == 1),
        'is_last_floor':   int(p.stock == p.stock_total),
        'hist_district':   p.hist_district,
        'wall':            p.wall,
        'project':         p.project,
    }
    return pd.DataFrame([row])[ALL_FEATURES]

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'RealtyIQ ML'}

@app.post('/predict')
def predict(p: PropertyInput):
    try:
        X_df  = build_features(p)
        X_t   = preprocessor.transform(X_df)

        # Прогноз (log-scale -> оригінальний масштаб)
        log_pred   = model.predict(X_t)[0]
        price_sm   = float(np.expm1(log_pred))

        # Довірчий інтервал (на основі квантильної регресії з навчання)
        # Використовуємо відносну ширину з збережених результатів
        avg_width_pct = interval_meta['quantile_avg_width'] / 1500
        ci_lower_95 = price_sm * (1 - avg_width_pct * 0.6)
        ci_upper_95 = price_sm * (1 + avg_width_pct * 0.6)
        ci_lower_80 = price_sm * (1 - avg_width_pct * 0.35)
        ci_upper_80 = price_sm * (1 + avg_width_pct * 0.35)

        # SHAP значення
        shap_vals  = explainer.shap_values(X_t)[0]
        shap_out   = []
        for feat, sv, fv in zip(ALL_FEATURES, shap_vals, X_t[0]):
            shap_out.append({
                'feature':    NAME_MAP.get(feat, feat),
                'shap_value': round(float(sv), 5),
                'value':      round(float(fv), 3),
            })
        shap_out.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        return {
            'predicted_price_sm': round(price_sm, 2),
            'predicted_total':    round(price_sm * p.area_total, 2),
            'ci_lower_95':        round(ci_lower_95, 2),
            'ci_upper_95':        round(ci_upper_95, 2),
            'ci_lower_80':        round(ci_lower_80, 2),
            'ci_upper_80':        round(ci_upper_80, 2),
            'shap_values':        shap_out[:10],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run('predict:app', host='0.0.0.0', port=8000, reload=False)
