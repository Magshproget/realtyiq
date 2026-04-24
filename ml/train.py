# -*- coding: utf-8 -*-
"""
train.py — Навчання та порівняння ML-моделей
Генерує графіки для Розділу 4.4 дипломної роботи

Покращення v2:
  1. Log-трансформація цільової змінної (price_sm -> log(price_sm))
  2. Розширена інженерія ознак
  3. Підібрані гіперпараметри XGBoost
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ── Шляхи ────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, 'data', 'class_flat_clean.csv')
PLOTS = os.path.join(BASE, 'ml', 'plots')
MODEL = os.path.join(BASE, 'ml', 'model')
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'figure.dpi': 150})

# ── 1. ЗАВАНТАЖЕННЯ ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA, encoding='utf-8')
print(f"[1] Завантажено: {df.shape[0]} записів")

# ── 2. РОЗШИРЕНА ІНЖЕНЕРІЯ ОЗНАК ──────────────────────────────────────────────
df['room_area']       = df['area_total'] / df['rooms'].clip(1)
df['kitchen_ratio']   = df['area_kitchen'] / df['area_total']
df['living_ratio']    = df['area_living']  / df['area_total']
df['is_first_floor']  = (df['stock'] == 1).astype(int)
df['is_last_floor']   = (df['stock'] == df['stock_total']).astype(int)
df['building_age']    = 2024 - df['year']

NUM_FEATURES = [
    'area_total', 'area_living', 'area_kitchen',
    'rooms', 'stock', 'stock_total', 'year', 'building_age',
    'floor_ratio', 'is_new_building', 'is_complex',
    'room_area', 'kitchen_ratio', 'living_ratio',
    'is_first_floor', 'is_last_floor'
]
CAT_FEATURES = ['hist_district', 'wall', 'project']
TARGET       = 'price_sm'

df = df[NUM_FEATURES + CAT_FEATURES + [TARGET]].dropna(subset=[TARGET])
X  = df[NUM_FEATURES + CAT_FEATURES]
y  = df[TARGET]

# Log-трансформація: нормалізує розподіл цін, знижує вплив викидів
y_log = np.log1p(y)

print(f"[2] Ознак: {len(NUM_FEATURES)} числових + {len(CAT_FEATURES)} категоріальних")
print(f"    Цільова: log(price_sm), mean={y_log.mean():.3f}, std={y_log.std():.3f}")

# ── 3. РОЗБИВКА ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test_orig  = np.expm1(y_test)
print(f"[3] Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. ПРЕПРОЦЕСИНГ ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_FEATURES),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1), CAT_FEATURES)
])

# ── 5. МОДЕЛІ ─────────────────────────────────────────────────────────────────
models = {
    'Лінійна регресія':  LinearRegression(),
    'Ridge регресія':    Ridge(alpha=10),
    'Дерево рішень':     DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest':     RandomForestRegressor(
                             n_estimators=300, max_depth=14,
                             min_samples_leaf=3, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(
                             n_estimators=300, max_depth=5,
                             learning_rate=0.05, random_state=42),
    'XGBoost':           XGBRegressor(
                             n_estimators=500, max_depth=6,
                             learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3,
                             reg_alpha=0.1, reg_lambda=1.0,
                             random_state=42, verbosity=0),
    'LightGBM':          LGBMRegressor(
                             n_estimators=500, max_depth=7,
                             learning_rate=0.03, num_leaves=63,
                             min_child_samples=20, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, verbose=-1),
    'Stacking (XGB+LGB+RF)': StackingRegressor(
                             estimators=[
                                 ('xgb', XGBRegressor(n_estimators=300, max_depth=6,
                                     learning_rate=0.03, subsample=0.8,
                                     colsample_bytree=0.8, random_state=42, verbosity=0)),
                                 ('lgb', LGBMRegressor(n_estimators=300, max_depth=7,
                                     learning_rate=0.03, num_leaves=63,
                                     random_state=42, verbose=-1)),
                                 ('rf',  RandomForestRegressor(n_estimators=200,
                                     max_depth=14, random_state=42, n_jobs=-1)),
                             ],
                             final_estimator=Ridge(alpha=1.0),
                             cv=5, n_jobs=-1),
}

# ── 6. НАВЧАННЯ ───────────────────────────────────────────────────────────────
print("\n[4] Навчання моделей (log-space)...")
results = []

for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)

    y_pred_log  = pipe.predict(X_test)
    y_pred_orig = np.expm1(y_pred_log)

    mae  = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2   = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    results.append({'Модель': name, 'MAE': mae, 'RMSE': rmse,
                    'R2': r2, 'MAPE': mape})
    print(f"  {name:<22}  MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.4f}  MAPE={mape:.1f}%")

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)

# ── 7. ЗБЕРЕЖЕННЯ НАЙКРАЩОЇ МОДЕЛІ ───────────────────────────────────────────
best_pipe = Pipeline([('pre', preprocessor),
                      ('model', XGBRegressor(
                          n_estimators=500, max_depth=6,
                          learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=3,
                          reg_alpha=0.1, reg_lambda=1.0,
                          random_state=42, verbosity=0))])
best_pipe.fit(X_train, y_train)

with open(os.path.join(MODEL, 'xgboost_model.pkl'), 'wb') as f:
    pickle.dump(best_pipe, f)
with open(os.path.join(MODEL, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(NUM_FEATURES + CAT_FEATURES, f)
with open(os.path.join(MODEL, 'num_features.pkl'), 'wb') as f:
    pickle.dump(NUM_FEATURES, f)

print(f"\n[OK] Модель збережено: ml/model/xgboost_model.pkl")

y_pred_best = np.expm1(best_pipe.predict(X_test))

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 1 — Порівняння R2
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#E74C3C' if r < 0.7 else '#F39C12' if r < 0.85 else '#2ECC71'
          for r in results_df['R2']]
bars = ax.barh(results_df['Модель'], results_df['R2'],
               color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, results_df['R2']):
    ax.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='right',
            color='white', fontweight='bold', fontsize=10)
ax.set_xlabel('Коефіцієнт детермінації R2')
ax.set_title('Рисунок 4.9 — Порівняння моделей за метрикою R2')
ax.set_xlim(0, 1.05)
ax.axvline(0.85, color='gray', linestyle='--', alpha=0.5, label='Поріг якості (0.85)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '09_models_r2.png'), bbox_inches='tight')
plt.close()
print("[5] Збережено: 09_models_r2.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 2 — MAE та RMSE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
palette = sns.color_palette('Set2', len(results_df))

axes[0].barh(results_df['Модель'], results_df['MAE'],
             color=palette, edgecolor='white', height=0.6)
axes[0].set_xlabel('MAE (USD/m2)')
axes[0].set_title('MAE — середня абсолютна похибка')
for i, v in enumerate(results_df['MAE']):
    axes[0].text(v + 1, i, f'{v:.0f}', va='center', fontsize=9)

axes[1].barh(results_df['Модель'], results_df['RMSE'],
             color=palette, edgecolor='white', height=0.6)
axes[1].set_xlabel('RMSE (USD/m2)')
axes[1].set_title('RMSE — середньоквадратична похибка')
for i, v in enumerate(results_df['RMSE']):
    axes[1].text(v + 1, i, f'{v:.0f}', va='center', fontsize=9)

fig.suptitle('Рисунок 4.10 — Порівняння моделей за метриками MAE та RMSE',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '10_models_mae_rmse.png'), bbox_inches='tight')
plt.close()
print("[6] Збережено: 10_models_mae_rmse.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 3 — Реальні vs Прогнозні
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test_orig, y_pred_best, alpha=0.25, s=8, color='#3498DB')
lims = [min(y_test_orig.min(), y_pred_best.min()),
        max(y_test_orig.max(), y_pred_best.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ідеальний прогноз')
ax.set_xlabel('Реальна ціна, USD/m2')
ax.set_ylabel('Прогнозована ціна, USD/m2')
ax.set_title('Рисунок 4.11 — Реальні vs прогнозні значення (XGBoost)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '11_actual_vs_predicted.png'), bbox_inches='tight')
plt.close()
print("[7] Збережено: 11_actual_vs_predicted.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 4 — Залишки
# ══════════════════════════════════════════════════════════════════════════════
residuals = y_test_orig - y_pred_best
pct_errors = (residuals / y_test_orig) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(pct_errors, bins=60, color='#9B59B6', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_xlabel('Відносна похибка, %')
axes[0].set_ylabel('Кількість')
axes[0].set_title('Розподіл відносних похибок')
axes[0].set_xlim(-60, 60)

axes[1].scatter(y_pred_best, residuals, alpha=0.2, s=8, color='#9B59B6')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Прогнозована ціна, USD/m2')
axes[1].set_ylabel('Залишки')
axes[1].set_title('Залишки vs Прогнозовані значення')

fig.suptitle('Рисунок 4.12 — Аналіз залишків моделі XGBoost',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '12_residuals.png'), bbox_inches='tight')
plt.close()
print("[8] Збережено: 12_residuals.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 5 — Крос-валідація топ-3
# ══════════════════════════════════════════════════════════════════════════════
top3 = {
    'Random Forest': Pipeline([('pre', preprocessor),
                               ('model', RandomForestRegressor(
                                   n_estimators=200, max_depth=14,
                                   min_samples_leaf=3, random_state=42, n_jobs=-1))]),
    'XGBoost':       Pipeline([('pre', preprocessor),
                               ('model', XGBRegressor(
                                   n_estimators=300, max_depth=6, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbosity=0))]),
    'LightGBM':      Pipeline([('pre', preprocessor),
                               ('model', LGBMRegressor(
                                   n_estimators=300, max_depth=7, learning_rate=0.03,
                                   num_leaves=63, random_state=42, verbose=-1))]),
}

print("\n[9] Крос-валідація (5-fold)...")
cv_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in top3.items():
    scores = cross_val_score(pipe, X, y_log, cv=kf, scoring='r2', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<15}  R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
clrs = ['#2ECC71', '#3498DB', '#E67E22']
for i, (name, scores) in enumerate(cv_results.items()):
    ax.boxplot(scores, positions=[i], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor=clrs[i], alpha=0.7))
ax.set_xticks(range(len(cv_results)))
ax.set_xticklabels(cv_results.keys())
ax.set_ylabel('R2 (5-fold CV, log-space)')
ax.set_title('Рисунок 4.13 — Крос-валідація топ-3 моделей')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '13_cross_validation.png'), bbox_inches='tight')
plt.close()
print("[9] Збережено: 13_cross_validation.png")

# ── ПІДСУМОК ──────────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("ЗВЕДЕНА ТАБЛИЦЯ МЕТРИК (оригінальний масштаб цін)")
print("="*68)
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print("="*68)
print(f"\n[OK] Всі графіки: ml/plots/")
