# -*- coding: utf-8 -*-
"""
predict.py — FastAPI ML-сервiс
Приймає параметри квартири, повертає прогноз + SHAP + iнтервали
"""

import os
import pickle
import json
import math
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsRegressor
import uvicorn

# -- Шляхи -------------------------------------------------------------------
BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE, 'ml', 'model', 'xgboost_model.pkl')
INTERVAL_PATH = os.path.join(BASE, 'ml', 'model', 'interval_results.json')
ENRICHED_CSV  = os.path.join(BASE, 'data', 'class_flat_enriched.csv')

# -- Завантаження моделі -----------------------------------------------------
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

with open(INTERVAL_PATH, 'r') as f:
    interval_meta = json.load(f)

preprocessor   = pipeline.named_steps['pre']
stacking_model = pipeline.named_steps['model']
xgb_model      = stacking_model.named_estimators_["xgb"]   # XGBRegressor з Stacking
explainer      = shap.TreeExplainer(xgb_model)

# -- Завантаження тренувальних даних для KNN price lag -----------------------
_df = pd.read_csv(ENRICHED_CSV, encoding='utf-8')
_df['room_area']      = _df['area_total'] / _df['rooms'].clip(1)
_df['kitchen_ratio']  = _df['area_kitchen'] / _df['area_total']
_df['living_ratio']   = _df['area_living']  / _df['area_total']
_df['is_first_floor'] = (_df['stock'] == 1).astype(int)
_df['is_last_floor']  = (_df['stock'] == _df['stock_total']).astype(int)
_df['building_age']   = 2024 - _df['year']
for _c in ['dist_center','dist_metro','dist_park','dist_school','dist_supermarket','dist_hospital']:
    if _c in _df.columns:
        _df[f'log_{_c}'] = np.log1p(_df[_c])
_df['metro_rightbank'] = _df['dist_metro'] * _df['is_right_bank']
_df['center_area']     = _df['dist_center'] * _df['area_total']

_lat_med = float(_df['lat'].median())
_lon_med = float(_df['lon'].median())
_coords  = _df[['lat','lon']].fillna({'lat': _lat_med, 'lon': _lon_med}).values
_prices  = _df['price_sm'].values

_knn = KNeighborsRegressor(n_neighbors=16, weights='distance',
                            metric='haversine', algorithm='ball_tree')
_knn.fit(np.radians(_coords), _prices)
_, _knn_idxs = _knn.kneighbors(np.radians(_coords), n_neighbors=16)
_df['knn_price_lag'] = [np.median(_prices[_knn_idxs[i,1:]]) for i in range(len(_prices))]

# Медіани просторових фіч по районах (fallback якщо lat/lon невідомі)
_SPATIAL_COLS = ['dist_center','dist_metro','is_right_bank','lat','lon',
                 'dist_park','dist_school','dist_supermarket','dist_hospital',
                 'log_dist_center','log_dist_metro','log_dist_park','log_dist_school',
                 'log_dist_supermarket','log_dist_hospital','metro_rightbank','center_area',
                 'knn_price_lag']
_district_medians = _df.groupby('hist_district')[_SPATIAL_COLS].median()
_global_medians   = _df[_SPATIAL_COLS].median()

NUM_FEATURES = [
    'area_total','area_living','area_kitchen','rooms','stock','stock_total',
    'year','building_age','floor_ratio','is_new_building','is_complex',
    'room_area','kitchen_ratio','living_ratio','is_first_floor','is_last_floor',
    'dist_center','dist_metro','is_right_bank','lat','lon',
    'dist_park','dist_school','dist_supermarket','dist_hospital',
    'log_dist_center','log_dist_metro','log_dist_park','log_dist_school',
    'log_dist_supermarket','log_dist_hospital',
    'metro_rightbank','center_area','knn_price_lag',
]
CAT_FEATURES  = ['hist_district', 'wall', 'project']
ALL_FEATURES  = NUM_FEATURES + CAT_FEATURES

KHRESCHATYK = (50.4501, 30.5234)

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2
         + math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon2-lon1)/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

NAME_MAP = {
    'area_total':           'Загальна площа',
    'area_living':          'Житлова площа',
    'area_kitchen':         'Площа кухні',
    'rooms':                'К-сть кімнат',
    'stock':                'Поверх',
    'stock_total':          'Поверховість будинку',
    'year':                 'Рік побудови',
    'building_age':         'Вік будинку',
    'floor_ratio':          'Відносний поверх',
    'is_new_building':      'Нова забудова',
    'is_complex':           'Належить до ЖК',
    'room_area':            'Площа на кімнату',
    'kitchen_ratio':        'Частка кухні',
    'living_ratio':         'Частка житлової площі',
    'is_first_floor':       'Перший поверх',
    'is_last_floor':        'Останній поверх',
    'dist_center':          'Відстань до центру',
    'dist_metro':           'Відстань до метро',
    'is_right_bank':        'Правий берег',
    'lat':                  'Широта',
    'lon':                  'Довгота',
    'dist_park':            'Відстань до парку',
    'dist_school':          'Відстань до школи',
    'dist_supermarket':     'Відстань до супермаркету',
    'dist_hospital':        'Відстань до лікарні',
    'log_dist_center':      'Ln(відстань до центру)',
    'log_dist_metro':       'Ln(відстань до метро)',
    'log_dist_park':        'Ln(відстань до парку)',
    'log_dist_school':      'Ln(відстань до школи)',
    'log_dist_supermarket': 'Ln(відстань до супермаркету)',
    'log_dist_hospital':    'Ln(відстань до лікарні)',
    'metro_rightbank':      'Метро × правий берег',
    'center_area':          'Центр × площа',
    'knn_price_lag':        'Медіана цін сусідів',
    'hist_district':        'Історичний район',
    'wall':                 'Матеріал стін',
    'project':              'Тип проекту',
}

# -- FastAPI -----------------------------------------------------------------
app = FastAPI(title='RealtyIQ ML Service', version='2.0')
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
    lat:           float  = None
    lon:           float  = None

def _get_spatial_fallback(district: str) -> dict:
    if district in _district_medians.index:
        row = _district_medians.loc[district]
    else:
        row = _global_medians
    return row.to_dict()

def build_features(p: PropertyInput) -> pd.DataFrame:
    area_living  = p.area_living  or p.area_total * 0.6
    area_kitchen = p.area_kitchen or 10.0

    has_coords = (p.lat is not None and p.lon is not None
                  and not math.isnan(p.lat) and not math.isnan(p.lon))

    spatial = _get_spatial_fallback(p.hist_district)

    if has_coords:
        lat, lon = p.lat, p.lon
        dist_center = haversine(lat, lon, *KHRESCHATYK)
        is_right_bank = int(lon < 30.54)
        # KNN price lag
        pt = np.radians([[lat, lon]])
        _, idxs = _knn.kneighbors(pt, n_neighbors=16)
        knn_price_lag = float(np.median(_prices[idxs[0]]))

        spatial.update({
            'lat': lat, 'lon': lon,
            'dist_center': dist_center,
            'is_right_bank': is_right_bank,
            'knn_price_lag': knn_price_lag,
            'log_dist_center': math.log1p(dist_center),
            'center_area': dist_center * p.area_total,
        })

    floor_ratio = p.stock / max(p.stock_total, 1)

    row = {
        'area_total':      p.area_total,
        'area_living':     area_living,
        'area_kitchen':    area_kitchen,
        'rooms':           p.rooms,
        'stock':           p.stock,
        'stock_total':     p.stock_total,
        'year':            p.year,
        'building_age':    2024 - p.year,
        'floor_ratio':     floor_ratio,
        'is_new_building': int(p.year >= 2010),
        'is_complex':      p.is_complex,
        'room_area':       p.area_total / max(p.rooms, 1),
        'kitchen_ratio':   area_kitchen / p.area_total,
        'living_ratio':    area_living  / p.area_total,
        'is_first_floor':  int(p.stock == 1),
        'is_last_floor':   int(p.stock == p.stock_total),
        'dist_center':     spatial.get('dist_center', _global_medians['dist_center']),
        'dist_metro':      spatial.get('dist_metro',  _global_medians['dist_metro']),
        'is_right_bank':   spatial.get('is_right_bank', _global_medians['is_right_bank']),
        'lat':             spatial.get('lat',  _lat_med),
        'lon':             spatial.get('lon',  _lon_med),
        'dist_park':       spatial.get('dist_park', _global_medians['dist_park']),
        'dist_school':     spatial.get('dist_school', _global_medians['dist_school']),
        'dist_supermarket':spatial.get('dist_supermarket', _global_medians['dist_supermarket']),
        'dist_hospital':   spatial.get('dist_hospital', _global_medians['dist_hospital']),
        'log_dist_center': spatial.get('log_dist_center', _global_medians['log_dist_center']),
        'log_dist_metro':  spatial.get('log_dist_metro',  _global_medians['log_dist_metro']),
        'log_dist_park':   spatial.get('log_dist_park',   _global_medians['log_dist_park']),
        'log_dist_school': spatial.get('log_dist_school', _global_medians['log_dist_school']),
        'log_dist_supermarket': spatial.get('log_dist_supermarket', _global_medians['log_dist_supermarket']),
        'log_dist_hospital':    spatial.get('log_dist_hospital',    _global_medians['log_dist_hospital']),
        'metro_rightbank': spatial.get('dist_metro', _global_medians['dist_metro']) * spatial.get('is_right_bank', _global_medians['is_right_bank']),
        'center_area':     spatial.get('dist_center', _global_medians['dist_center']) * p.area_total,
        'knn_price_lag':   spatial.get('knn_price_lag', _global_medians['knn_price_lag']),
        'hist_district':   p.hist_district,
        'wall':            p.wall,
        'project':         p.project,
    }
    return pd.DataFrame([row])[ALL_FEATURES]

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'RealtyIQ ML v2'}

@app.post('/predict')
def predict(p: PropertyInput):
    try:
        X_df = build_features(p)
        X_t  = preprocessor.transform(X_df)

        log_pred = pipeline.predict(X_df)[0]
        price_sm = float(np.expm1(log_pred))

        # Довірчий інтервал (Bootstrap 95% CI ширина зі збережених результатів)
        avg_width = interval_meta.get('bootstrap_avg_width', price_sm * 0.4)
        ci_lower_95 = max(0, price_sm - avg_width / 2)
        ci_upper_95 = price_sm + avg_width / 2
        ci_lower_80 = max(0, price_sm - avg_width * 0.3)
        ci_upper_80 = price_sm + avg_width * 0.3

        # SHAP значення (XGB base model)
        shap_vals = explainer.shap_values(X_t)[0]
        shap_out  = []
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
