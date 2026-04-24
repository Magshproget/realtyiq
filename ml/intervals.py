# -*- coding: utf-8 -*-
"""
intervals.py — Iнтервальне оцiнювання та моделювання ризикiв
Генерує графiки для Роздiлу 4.6 дипломної роботи

Вимога наукового керiвника:
  - Iмовiрнiснi пiдходи та iнтервальне оцiнювання
  - Моделювання ризикiв на волатильному ринку
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# -- Шляхи -------------------------------------------------------------------
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, 'data', 'class_flat_clean.csv')
PLOTS = os.path.join(BASE, 'ml', 'plots')
MODEL = os.path.join(BASE, 'ml', 'model')
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

df['room_area']      = df['area_total'] / df['rooms'].clip(1)
df['kitchen_ratio']  = df['area_kitchen'] / df['area_total']
df['living_ratio']   = df['area_living']  / df['area_total']
df['is_first_floor'] = (df['stock'] == 1).astype(int)
df['is_last_floor']  = (df['stock'] == df['stock_total']).astype(int)
df['building_age']   = 2024 - df['year']

df = df[ALL_FEATURES + ['price_sm']].dropna(subset=['price_sm'])
X  = df[ALL_FEATURES]
y  = np.log1p(df['price_sm'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_FEATURES),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1), CAT_FEATURES)
])

print("[1] Дані завантажено")

# ============================================================================
# МЕТОД 1: Bootstrap — оцінка невизначеності через повторне навчання
# ============================================================================
print("[2] Bootstrap (50 ітерацій)...")

N_BOOTSTRAP = 50
rng         = np.random.RandomState(42)
boot_preds  = []

X_train_t = preprocessor.fit_transform(X_train, y_train)
X_test_t  = preprocessor.transform(X_test)

for i in range(N_BOOTSTRAP):
    idx = rng.choice(len(X_train_t), len(X_train_t), replace=True)
    model = XGBRegressor(n_estimators=200, max_depth=6,
                         learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, random_state=i, verbosity=0)
    model.fit(X_train_t[idx], y_train.iloc[idx])
    boot_preds.append(np.expm1(model.predict(X_test_t)))
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{N_BOOTSTRAP}...")

boot_preds  = np.array(boot_preds)           # (50, n_test)
y_test_orig = np.expm1(y_test)

pred_mean   = boot_preds.mean(axis=0)
pred_std    = boot_preds.std(axis=0)
ci_lower_95 = np.percentile(boot_preds, 2.5,  axis=0)
ci_upper_95 = np.percentile(boot_preds, 97.5, axis=0)
ci_lower_80 = np.percentile(boot_preds, 10.0, axis=0)
ci_upper_80 = np.percentile(boot_preds, 90.0, axis=0)

coverage_95 = np.mean((y_test_orig >= ci_lower_95) & (y_test_orig <= ci_upper_95))
coverage_80 = np.mean((y_test_orig >= ci_lower_80) & (y_test_orig <= ci_upper_80))
avg_width   = (ci_upper_95 - ci_lower_95).mean()

print(f"   Покриття 95%: {coverage_95:.1%}")
print(f"   Покриття 80%: {coverage_80:.1%}")
print(f"   Середня ширина 95% CI: {avg_width:.0f} USD/m2")

# ============================================================================
# МЕТОД 2: Квантильна регресія (XGBoost objective='reg:quantileerror')
# ============================================================================
print("[3] Квантильна регресія (q=0.05, 0.5, 0.95)...")

quantile_preds = {}
for q in [0.05, 0.5, 0.95]:
    qmodel = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:quantileerror',
        quantile_alpha=q, random_state=42, verbosity=0
    )
    qmodel.fit(X_train_t, y_train)
    quantile_preds[q] = np.expm1(qmodel.predict(X_test_t))

q_lower = quantile_preds[0.05]
q_median = quantile_preds[0.5]
q_upper  = quantile_preds[0.95]

coverage_q = np.mean((y_test_orig >= q_lower) & (y_test_orig <= q_upper))
width_q    = (q_upper - q_lower).mean()
print(f"   Покриття 90% PI: {coverage_q:.1%}")
print(f"   Середня ширина PI: {width_q:.0f} USD/m2")

# ============================================================================
# ГРАФІК 1 — Bootstrap довірчі інтервали (перші 80 об'єктів)
# ============================================================================
n_show  = 80
sort_idx = np.argsort(y_test_orig.values)[:n_show]

fig, ax = plt.subplots(figsize=(14, 5))
x_pos = np.arange(n_show)

ax.fill_between(x_pos,
                ci_lower_95[sort_idx], ci_upper_95[sort_idx],
                alpha=0.25, color='#3498DB', label='95% довiрчий iнтервал')
ax.fill_between(x_pos,
                ci_lower_80[sort_idx], ci_upper_80[sort_idx],
                alpha=0.4,  color='#3498DB', label='80% довiрчий iнтервал')
ax.plot(x_pos, pred_mean[sort_idx],
        color='#2980B9', linewidth=1.5, label='Середнiй прогноз')
ax.scatter(x_pos, y_test_orig.values[sort_idx],
           color='#E74C3C', s=12, zorder=5, label='Реальна цiна')

ax.set_xlabel('Об\'єкт (впорядкований за зростанням цiни)')
ax.set_ylabel('Цiна за м2, USD')
ax.set_title('Рисунок 4.18 — Bootstrap довiрчi iнтервали прогнозу (80 об\'єктiв)')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '18_bootstrap_intervals.png'), bbox_inches='tight')
plt.close()
print("[4] Збережено: 18_bootstrap_intervals.png")

# ============================================================================
# ГРАФІК 2 — Квантильна регресія
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(x_pos,
                q_lower[sort_idx], q_upper[sort_idx],
                alpha=0.25, color='#27AE60', label='90% предикцiйний iнтервал (Q5-Q95)')
ax.plot(x_pos, q_median[sort_idx],
        color='#1E8449', linewidth=1.5, label='Медiанний прогноз (Q50)')
ax.scatter(x_pos, y_test_orig.values[sort_idx],
           color='#E74C3C', s=12, zorder=5, label='Реальна цiна')

ax.set_xlabel('Об\'єкт (впорядкований за зростанням цiни)')
ax.set_ylabel('Цiна за м2, USD')
ax.set_title('Рисунок 4.19 — Квантильна регресiя: предикцiйнi iнтервали (Q5–Q95)')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '19_quantile_intervals.png'), bbox_inches='tight')
plt.close()
print("[5] Збережено: 19_quantile_intervals.png")

# ============================================================================
# ГРАФІК 3 — Аналіз ризику: розподіл ширини інтервалів по районах
# ============================================================================
df_test = X_test.copy()
df_test['ci_width']   = ci_upper_95 - ci_lower_95
df_test['pred_mean']  = pred_mean
df_test['actual']     = y_test_orig.values
df_test['risk_pct']   = df_test['ci_width'] / df_test['pred_mean'] * 100

top_districts = df_test['hist_district'].value_counts().head(7).index
df_risk = df_test[df_test['hist_district'].isin(top_districts)]

risk_by_district = df_risk.groupby('hist_district')['risk_pct'].median().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors  = ['#E74C3C' if v > 50 else '#F39C12' if v > 35 else '#2ECC71'
           for v in risk_by_district.values]
bars    = ax.bar(risk_by_district.index, risk_by_district.values,
                 color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, risk_by_district.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Медiанна ширина CI / Прогноз, %')
ax.set_title('Рисунок 4.20 — Ризик прогнозу по районах (вiдносна ширина 95% CI)')
ax.set_xticklabels(risk_by_district.index, rotation=20, ha='right')
patches = [
    mpatches.Patch(color='#E74C3C', label='Високий ризик (>50%)'),
    mpatches.Patch(color='#F39C12', label='Середнiй ризик (35-50%)'),
    mpatches.Patch(color='#2ECC71', label='Низький ризик (<35%)'),
]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '20_risk_by_district.png'), bbox_inches='tight')
plt.close()
print("[6] Збережено: 20_risk_by_district.png")

# ============================================================================
# ГРАФІК 4 — Порівняння Bootstrap vs Quantile (зведений)
# ============================================================================
methods = ['Bootstrap\n95% CI', 'Quantile\n90% PI']
coverages = [coverage_95 * 100, coverage_q * 100]
widths    = [avg_width, width_q]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(methods, coverages, color=['#3498DB', '#27AE60'],
            edgecolor='white', width=0.5)
axes[0].axhline(95, color='#3498DB', linestyle='--', alpha=0.6, label='Цiльове 95%')
axes[0].axhline(90, color='#27AE60', linestyle='--', alpha=0.6, label='Цiльове 90%')
axes[0].set_ylabel('Покриття, %')
axes[0].set_title('Фактичне покриття')
axes[0].set_ylim(0, 110)
for i, v in enumerate(coverages):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0].legend(fontsize=8)

axes[1].bar(methods, widths, color=['#3498DB', '#27AE60'],
            edgecolor='white', width=0.5)
axes[1].set_ylabel('Середня ширина iнтервалу, USD/m2')
axes[1].set_title('Ширина iнтервалу (менше = точнiше)')
for i, v in enumerate(widths):
    axes[1].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')

fig.suptitle('Рисунок 4.21 — Порiвняння методiв iнтервального оцiнювання',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '21_methods_comparison.png'), bbox_inches='tight')
plt.close()
print("[7] Збережено: 21_methods_comparison.png")

# -- ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ --------------------------------------------------
results = {
    'bootstrap_coverage_95': float(coverage_95),
    'bootstrap_coverage_80': float(coverage_80),
    'bootstrap_avg_width':   float(avg_width),
    'quantile_coverage_90':  float(coverage_q),
    'quantile_avg_width':    float(width_q),
}
import json
with open(os.path.join(MODEL, 'interval_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n--- Зведенi результати ---")
print(f"  Bootstrap 95% CI  покриття: {coverage_95:.1%}   ширина: {avg_width:.0f} USD/m2")
print(f"  Bootstrap 80% CI  покриття: {coverage_80:.1%}   ширина: {(ci_upper_80-ci_lower_80).mean():.0f} USD/m2")
print(f"  Quantile  90% PI  покриття: {coverage_q:.1%}   ширина: {width_q:.0f} USD/m2")
print(f"\n[OK] Всi графiки: ml/plots/")
