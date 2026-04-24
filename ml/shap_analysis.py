# -*- coding: utf-8 -*-
"""
shap_analysis.py — Аналiз чутливостi моделi (SHAP)
Генерує графiки для Роздiлу 4.5 дипломної роботи

Вимога наукового керiвника:
  - Sensitivity Analysis: оцiнка впливу змiни параметрiв на прогноз
  - Кiлькiсна оцiнка внеску кожної ознаки через апарат SHAP
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

warnings.filterwarnings('ignore')

# -- Шляхи -------------------------------------------------------------------
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, 'data', 'class_flat_clean.csv')
MODEL = os.path.join(BASE, 'ml', 'model', 'xgboost_model.pkl')
PLOTS = os.path.join(BASE, 'ml', 'plots')
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'figure.dpi': 150})

# -- 1. ЗАВАНТАЖЕННЯ ----------------------------------------------------------
df = pd.read_csv(DATA, encoding='utf-8')

NUM_FEATURES = [
    'area_total', 'area_living', 'area_kitchen',
    'rooms', 'stock', 'stock_total', 'year', 'building_age',
    'floor_ratio', 'is_new_building', 'is_complex',
    'room_area', 'kitchen_ratio', 'living_ratio',
    'is_first_floor', 'is_last_floor'
]
CAT_FEATURES = ['hist_district', 'wall', 'project']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Розраховуємо похідні ознаки
df['room_area']      = df['area_total'] / df['rooms'].clip(1)
df['kitchen_ratio']  = df['area_kitchen'] / df['area_total']
df['living_ratio']   = df['area_living']  / df['area_total']
df['is_first_floor'] = (df['stock'] == 1).astype(int)
df['is_last_floor']  = (df['stock'] == df['stock_total']).astype(int)
df['building_age']   = 2024 - df['year']

df = df[ALL_FEATURES + ['price_sm']].dropna(subset=['price_sm'])
X  = df[ALL_FEATURES]

with open(MODEL, 'rb') as f:
    pipeline = pickle.load(f)

preprocessor = pipeline.named_steps['pre']
model        = pipeline.named_steps['model']

print("[1] Модель та дані завантажено")

# -- 2. ТРАНСФОРМАЦІЯ ОЗНАК --------------------------------------------------
X_transformed = preprocessor.transform(X)

# Відновлюємо назви ознак після OrdinalEncoder
cat_names = []
for cat in CAT_FEATURES:
    cat_names.append(cat)

feature_names = NUM_FEATURES + cat_names
print(f"[2] Трансформовано: {X_transformed.shape[1]} ознак")

# -- 3. SHAP EXPLAINER --------------------------------------------------------
explainer   = shap.TreeExplainer(model)
# Беремо вибірку 1500 для швидкості
sample_idx  = np.random.RandomState(42).choice(len(X_transformed), 1500, replace=False)
X_sample    = X_transformed[sample_idx]
shap_values = explainer.shap_values(X_sample)

print("[3] SHAP values розраховано")

# ============================================================================
# ГРАФІК 1 — Summary Plot (bee swarm) — найважливіший для диплому
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_sample,
    feature_names=feature_names,
    show=False, plot_size=None,
    max_display=16
)
plt.title('Рисунок 4.14 — SHAP Summary Plot: вплив ознак на прогноз ціни',
          fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '14_shap_summary.png'), bbox_inches='tight')
plt.close()
print("[4] Збережено: 14_shap_summary.png")

# ============================================================================
# ГРАФІК 2 — Bar Plot (середній |SHAP|) — рейтинг важливості ознак
# ============================================================================
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_names
).sort_values(ascending=True)

# Перекладаємо назви для читабельності
name_map = {
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
mean_shap.index = [name_map.get(n, n) for n in mean_shap.index]

colors = ['#2ECC71' if v >= mean_shap.quantile(0.66)
          else '#F39C12' if v >= mean_shap.quantile(0.33)
          else '#E74C3C' for v in mean_shap.values]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(mean_shap.index, mean_shap.values,
               color=colors, edgecolor='white', height=0.7)
for bar, val in zip(bars, mean_shap.values):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5)
ax.set_xlabel('Середнє |SHAP| значення (вплив на log-ціну)')
ax.set_title('Рисунок 4.15 — Рейтинг важливості ознак (SHAP Feature Importance)')
patches = [
    mpatches.Patch(color='#2ECC71', label='Висока важливість'),
    mpatches.Patch(color='#F39C12', label='Середня важливість'),
    mpatches.Patch(color='#E74C3C', label='Низька важливість'),
]
ax.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '15_shap_importance.png'), bbox_inches='tight')
plt.close()
print("[5] Збережено: 15_shap_importance.png")

# ============================================================================
# ГРАФІК 3 — Waterfall plot для одного об'єкта (контрольний приклад)
# ============================================================================
# Беремо типову 2-кімнатну квартиру з датасету
typical_mask = (
    (df['rooms'] == 2) &
    (df['area_total'].between(45, 65)) &
    (df['hist_district'] != 'Невідомо')
)
typical_idx = df[typical_mask].index[0]
X_single    = X.loc[[typical_idx]]
X_single_t  = preprocessor.transform(X_single)

exp_single   = shap.TreeExplainer(model)
shap_single  = exp_single(X_single_t)

# Міняємо назви ознак
shap_single.feature_names = [name_map.get(n, n) for n in feature_names]

fig, ax = plt.subplots(figsize=(11, 6))
shap.waterfall_plot(shap_single[0], max_display=12, show=False)
plt.title('Рисунок 4.16 — SHAP Waterfall: пояснення прогнозу для конкретного об\'єкта',
          fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '16_shap_waterfall.png'), bbox_inches='tight')
plt.close()
print("[6] Збережено: 16_shap_waterfall.png")

# ============================================================================
# ГРАФІК 4 — Dependence plot: площа vs SHAP (аналіз чутливості)
# ============================================================================
area_idx = feature_names.index('area_total')
dist_idx = feature_names.index('hist_district')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sc1 = axes[0].scatter(
    X_sample[:, area_idx],
    shap_values[:, area_idx],
    c=X_sample[:, dist_idx], cmap='viridis',
    alpha=0.3, s=10
)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Загальна площа, м2')
axes[0].set_ylabel('SHAP значення')
axes[0].set_title('Залежність: Площа → SHAP')
plt.colorbar(sc1, ax=axes[0], label='Район (encoded)')

year_idx = feature_names.index('year')
sc2 = axes[1].scatter(
    X_sample[:, year_idx],
    shap_values[:, year_idx],
    c=X_sample[:, area_idx], cmap='plasma',
    alpha=0.3, s=10
)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Рік побудови')
axes[1].set_ylabel('SHAP значення')
axes[1].set_title('Залежність: Рік побудови -> SHAP')
plt.colorbar(sc2, ax=axes[1], label='Площа, м2')

fig.suptitle('Рисунок 4.17 — Аналiз чутливостi: залежнiсть SHAP-значень вiд ключових ознак',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '17_shap_dependence.png'), bbox_inches='tight')
plt.close()
print("[7] Збережено: 17_shap_dependence.png")

# -- ЗВЕДЕНІ РЕЗУЛЬТАТИ -------------------------------------------------------
print("\n--- Топ-5 найважливіших ознак (SHAP) ---")
top5 = mean_shap.sort_values(ascending=False).head(5)
for feat, val in top5.items():
    print(f"  {feat:<30}  SHAP={val:.4f}")

print(f"\n[OK] Всі SHAP графіки: ml/plots/")
