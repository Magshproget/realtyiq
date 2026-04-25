import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

models = [
    'Лінійна регресія',
    'Ridge регресія',
    'Дерево рішень',
    'Random Forest',
    'Gradient Boosting',
    'Stacking (XGB+LGB+RF)',
    'LightGBM',
    'XGBoost',
]

mae  = [436.7, 436.7, 331.9, 275.9, 283.6, 269.8, 264.8, 264.4]
rmse = [595.9, 596.0, 483.5, 401.7, 407.5, 388.0, 381.1, 379.8]
r2   = [0.220, 0.220, 0.486, 0.646, 0.635, 0.669, 0.681, 0.683]
mape = [25.6,  25.6,  19.7,  16.2,  16.4,  15.7,  15.3,  15.3]

colors = ['#e74c3c' if m > 20 else '#f39c12' if m > 17 else '#27ae60' for m in mape]

x = np.arange(len(models))
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# MAE
ax = axes[0, 0]
bars = ax.barh(models, mae, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, mae):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('MAE, USD/м²', fontsize=10)
ax.set_xlim(0, 680)
ax.axvline(264.4, color='#27ae60', linestyle='--', alpha=0.6, linewidth=1.5)
ax.invert_yaxis()

# RMSE
ax = axes[0, 1]
bars = ax.barh(models, rmse, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, rmse):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('RMSE, USD/м²', fontsize=10)
ax.set_xlim(0, 780)
ax.axvline(379.8, color='#27ae60', linestyle='--', alpha=0.6, linewidth=1.5)
ax.invert_yaxis()

# R2
ax = axes[1, 0]
bars = ax.barh(models, r2, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, r2):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('R²', fontsize=10)
ax.set_xlim(0, 0.85)
ax.axvline(0.683, color='#27ae60', linestyle='--', alpha=0.6, linewidth=1.5)
ax.invert_yaxis()

# MAPE
ax = axes[1, 1]
bars = ax.barh(models, mape, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, mape):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('%', fontsize=10)
ax.set_xlim(0, 34)
ax.axvline(15.3, color='#27ae60', linestyle='--', alpha=0.6, linewidth=1.5)
ax.invert_yaxis()

legend_patches = [
    mpatches.Patch(color='#27ae60', label='Найкращий результат (XGBoost)'),
    mpatches.Patch(color='#f39c12', label='Середній результат'),
    mpatches.Patch(color='#e74c3c', label='Слабкий результат'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])

out = os.path.join(os.path.dirname(__file__), 'plots', '22_models_comparison_diploma.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Збережено: {out}')
