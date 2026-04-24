# -*- coding: utf-8 -*-
"""
EDA — Розвідувальний аналіз датасету class_flat.csv
Генерує графіки для Розділу 4.1-4.2 дипломної роботи
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Шляхи ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(BASE, 'data', 'class_flat.csv')
OUTPUT = os.path.join(BASE, 'data', 'class_flat_clean.csv')
PLOTS  = os.path.join(BASE, 'eda', 'plots')
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── 1. ЗАВАНТАЖЕННЯ ───────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, encoding='utf-8')
print(f"[1] Завантажено: {df.shape[0]} рядків, {df.shape[1]} колонок")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 1 — Пропущені значення (до очищення)
# ══════════════════════════════════════════════════════════════════════════════
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(miss.index, miss.values, color='#E74C3C', edgecolor='white')
for bar, val in zip(bars, miss.values):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{val} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
ax.set_xlabel('Кількість пропущених значень')
ax.set_title('Рисунок 4.1 — Пропущені значення у вихідному датасеті')
ax.set_xlim(0, miss.max() * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '01_missing_values.png'), bbox_inches='tight')
plt.close()
print("[1] Збережено: 01_missing_values.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 2 — Розподіл price_sm ДО очищення
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['price_sm'], bins=80, color='#3498DB', edgecolor='white', alpha=0.8)
axes[0].set_title('До очищення (всі записи)')
axes[0].set_xlabel('Ціна за м², USD')
axes[0].set_ylabel('Кількість оголошень')
axes[0].axvline(df['price_sm'].median(), color='red', linestyle='--', label=f"Медіана: {df['price_sm'].median():.0f}")
axes[0].legend()

axes[1].hist(df['price_sm'], bins=80, color='#3498DB', edgecolor='white', alpha=0.8)
axes[1].set_xlim(0, 5000)
axes[1].set_title('До очищення (до 5000 USD/м²)')
axes[1].set_xlabel('Ціна за м², USD')
axes[1].set_ylabel('Кількість оголошень')

fig.suptitle('Рисунок 4.2 — Розподіл ціни за м² до очищення', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '02_price_dist_before.png'), bbox_inches='tight')
plt.close()
print("[2] Збережено: 02_price_dist_before.png")

# ══════════════════════════════════════════════════════════════════════════════
# ОЧИЩЕННЯ ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════
print("\n[ОЧИЩЕННЯ]")
n0 = len(df)

# 1. Прибираємо bad_proposal
df = df[df['bad_proposal'] == 0]
print(f"  Після фільтру bad_proposal:  {len(df)} рядків (- {n0 - len(df)})")

# 2. Лише USD (основна валюта ринку)
df = df[df['currency'] == 'USD']
print(f"  Після фільтру currency=USD:  {len(df)} рядків (- {n0 - len(df)})")

# 3. Прибираємо кімнати > 6 (комерція/помилки)
df = df[df['rooms'] <= 6]
print(f"  Після фільтру rooms <= 6:    {len(df)} рядків")

# 4. IQR-фільтр по price_sm
Q1 = df['price_sm'].quantile(0.05)
Q3 = df['price_sm'].quantile(0.95)
df = df[(df['price_sm'] >= Q1) & (df['price_sm'] <= Q3)]
print(f"  Після IQR (5%-95%):          {len(df)} рядків")

# 5. Площа: прибираємо нереальні значення
df = df[(df['area_total'] >= 15) & (df['area_total'] <= 250)]
print(f"  Після фільтру площі:         {len(df)} рядків")

# 6. Заповнення пропусків
df['year'] = df['year'].fillna(df['year'].median())
df['area_living'] = df['area_living'].fillna(df['area_total'] * 0.6)
df['area_kitchen'] = df['area_kitchen'].fillna(df['area_kitchen'].median())
df['stock_total'] = df['stock_total'].fillna(df['stock_total'].median())
df['wall'] = df['wall'].fillna('Невідомо')
df['project'] = df['project'].fillna('Невідомо')
df['hist_district'] = df['hist_district'].fillna('Невідомо')

# 7. Ознака "is_new_building"
median_year = df['year'].median()
df['is_new_building'] = (df['year'] >= 2000).astype(int)

# 8. Ознака "floor_ratio"
df['floor_ratio'] = df['stock'] / df['stock_total'].replace(0, 1)

# 9. is_complex
df['is_complex'] = df['complex'].notna().astype(int)

print(f"\n  Фінальний датасет: {len(df)} рядків, {df.shape[1]} колонок")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 3 — Розподіл price_sm ПІСЛЯ очищення
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df['price_sm'], bins=60, color='#2ECC71', edgecolor='white', alpha=0.85)
ax.axvline(df['price_sm'].mean(),   color='red',    linestyle='--', linewidth=1.5,
           label=f"Середнє: {df['price_sm'].mean():.0f}")
ax.axvline(df['price_sm'].median(), color='orange', linestyle='--', linewidth=1.5,
           label=f"Медіана: {df['price_sm'].median():.0f}")
ax.set_xlabel('Ціна за м², USD')
ax.set_ylabel('Кількість оголошень')
ax.set_title('Рисунок 4.3 — Розподіл ціни за м² після очищення')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '03_price_dist_after.png'), bbox_inches='tight')
plt.close()
print("[3] Збережено: 03_price_dist_after.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 4 — Ціна по районах (boxplot)
# ══════════════════════════════════════════════════════════════════════════════
top_districts = df['hist_district'].value_counts().head(8).index
df_dist = df[df['hist_district'].isin(top_districts)]

order = df_dist.groupby('hist_district')['price_sm'].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df_dist, x='hist_district', y='price_sm',
            order=order, palette='Blues_r', ax=ax,
            flierprops=dict(marker='.', markersize=2, alpha=0.3))
ax.set_xlabel('Район')
ax.set_ylabel('Ціна за м², USD')
ax.set_title('Рисунок 4.4 — Розподіл ціни за м² по районах міста')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '04_price_by_district.png'), bbox_inches='tight')
plt.close()
print("[4] Збережено: 04_price_by_district.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 5 — Ціна vs Площа (scatter)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(df['area_total'], df['price_sm'],
                     c=df['rooms'], cmap='viridis',
                     alpha=0.3, s=8)
plt.colorbar(scatter, ax=ax, label='Кількість кімнат')
ax.set_xlabel('Загальна площа, м²')
ax.set_ylabel('Ціна за м², USD')
ax.set_title('Рисунок 4.5 — Залежність ціни від площі (колір — кількість кімнат)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '05_area_vs_price.png'), bbox_inches='tight')
plt.close()
print("[5] Збережено: 05_area_vs_price.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 6 — Середня ціна по кількості кімнат
# ══════════════════════════════════════════════════════════════════════════════
rooms_avg = df.groupby('rooms')['price_sm'].median().reset_index()
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(rooms_avg['rooms'], rooms_avg['price_sm'],
              color='#9B59B6', edgecolor='white', width=0.6)
for bar, val in zip(bars, rooms_avg['price_sm']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Кількість кімнат')
ax.set_ylabel('Медіанна ціна за м², USD')
ax.set_title('Рисунок 4.6 — Медіанна ціна за м² залежно від кількості кімнат')
ax.set_xticks(rooms_avg['rooms'])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '06_rooms_vs_price.png'), bbox_inches='tight')
plt.close()
print("[6] Збережено: 06_rooms_vs_price.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 7 — Кореляційна матриця
# ══════════════════════════════════════════════════════════════════════════════
num_cols = ['price_sm', 'area_total', 'area_living', 'area_kitchen',
            'rooms', 'stock', 'stock_total', 'year', 'floor_ratio']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 9})
ax.set_title('Рисунок 4.7 — Кореляційна матриця числових ознак')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '07_correlation_matrix.png'), bbox_inches='tight')
plt.close()
print("[7] Збережено: 07_correlation_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 8 — Рік побудови vs ціна
# ══════════════════════════════════════════════════════════════════════════════
df_year = df[df['year'] >= 1950]
year_avg = df_year.groupby(df_year['year'].astype(int))['price_sm'].median()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(year_avg.index, year_avg.values, color='#E67E22', linewidth=2)
ax.fill_between(year_avg.index, year_avg.values, alpha=0.15, color='#E67E22')
ax.set_xlabel('Рік побудови')
ax.set_ylabel('Медіанна ціна за м², USD')
ax.set_title('Рисунок 4.8 — Медіанна ціна за м² залежно від року побудови')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '08_year_vs_price.png'), bbox_inches='tight')
plt.close()
print("[8] Збережено: 08_year_vs_price.png")

# ══════════════════════════════════════════════════════════════════════════════
# ЗБЕРЕЖЕННЯ ЧИСТОГО ДАТАСЕТУ
# ══════════════════════════════════════════════════════════════════════════════
df.to_csv(OUTPUT, index=False, encoding='utf-8')
print(f"\n[OK] Чистий датасет збережено: {OUTPUT}")
print(f"     Записів: {len(df)}")
print(f"\n[OK] Всі графіки в: {PLOTS}")

# ══════════════════════════════════════════════════════════════════════════════
# ЗВЕДЕНА СТАТИСТИКА
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Зведена статистика (чистий датасет) ───")
print(df[['price_sm','area_total','rooms','year']].describe().round(1).to_string())
