# -*- coding: utf-8 -*-
"""
Графік динаміки індексу цін на первинному ринку житла в Україні
Дані: minfin.com.ua (2015–2025, зростаючим підсумком, база = 100%)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Дані (апроксимація з графіка minfin.com.ua) ──────────────────────────────
years = [
    2015.92, 2016.17, 2016.42, 2016.67, 2016.92,
    2017.17, 2017.42, 2017.67, 2017.92,
    2018.17, 2018.42, 2018.67, 2018.92,
    2019.17, 2019.42, 2019.67, 2019.92,
    2020.17, 2020.42, 2020.67, 2020.92,
    2021.17, 2021.42, 2021.67, 2021.92,
    2022.17, 2022.42, 2022.67, 2022.92,
    2023.17, 2023.42, 2023.67, 2023.92,
    2024.17, 2024.42, 2024.67, 2024.92,
    2025.17, 2025.42, 2025.92,
]

# Загальний індекс цін (синя лінія)
general = [
    100, 97, 95, 96, 98,
    101, 104, 106, 108,
    111, 114, 116, 118,
    119, 121, 122, 124,
    126, 128, 130, 133,
    140, 150, 160, 168,
    170, 172, 175, 178,
    188, 198, 207, 213,
    220, 228, 236, 242,
    248, 252, 251,
]

# 1-кімнатні (фіолетова лінія — вища)
one_room = [
    100, 96, 94, 95, 97,
    100, 103, 106, 109,
    112, 115, 118, 121,
    122, 124, 125, 127,
    129, 131, 134, 138,
    146, 157, 167, 174,
    176, 178, 180, 182,
    193, 204, 213, 218,
    226, 234, 242, 249,
    255, 261, 264,
]

# 2-кімнатні (зелена лінія)
two_room = [
    100, 97, 95, 96, 98,
    101, 104, 106, 108,
    111, 113, 115, 117,
    118, 120, 121, 123,
    125, 127, 129, 132,
    139, 149, 158, 166,
    168, 170, 173, 176,
    186, 196, 205, 211,
    218, 226, 234, 240,
    246, 249, 250,
]

# 3-кімнатні (бірюзова лінія)
three_room = [
    100, 97, 95, 96, 98,
    101, 104, 105, 107,
    110, 112, 114, 116,
    117, 119, 120, 122,
    124, 126, 128, 131,
    138, 147, 156, 164,
    167, 169, 172, 175,
    184, 194, 203, 209,
    216, 224, 232, 238,
    244, 247, 250,
]

# ── Побудова графіка ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(years, one_room,  color='#8B5CF6', linewidth=1.8,
        label='1-кімнатні', zorder=4)
ax.plot(years, general,   color='#3B82F6', linewidth=2.2,
        label='Загальний індекс', zorder=5)
ax.plot(years, two_room,  color='#10B981', linewidth=1.8,
        label='2-кімнатні', zorder=3)
ax.plot(years, three_room,color='#06B6D4', linewidth=1.8,
        label='3-кімнатні', zorder=3)

# Виділення ключових подій
ax.axvline(2022.17, color='#EF4444', linestyle='--',
           linewidth=1.2, alpha=0.7, zorder=2)
ax.text(2022.22, 108, 'Початок\nвійни\n02.2022',
        fontsize=8, color='#EF4444', va='bottom')

ax.axvline(2020.17, color='#F59E0B', linestyle='--',
           linewidth=1.0, alpha=0.6, zorder=2)
ax.text(2020.22, 108, 'COVID-19\n2020',
        fontsize=8, color='#F59E0B', va='bottom')

# Горизонтальна лінія базового значення
ax.axhline(100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Фон
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')
ax.grid(axis='y', linestyle='--', alpha=0.35, color='gray')
ax.grid(axis='x', linestyle=':', alpha=0.2, color='gray')

# Осі
ax.set_xlim(2015.8, 2026.0)
ax.set_ylim(82, 280)
ax.set_xlabel('Рік', fontsize=11)
ax.set_ylabel('Індекс цін, % (база: грудень 2015 = 100%)', fontsize=11)

# X-тіки: лише роки
ax.set_xticks([2016, 2017, 2018, 2019, 2020,
               2021, 2022, 2023, 2024, 2025])
ax.set_xticklabels(['2016','2017','2018','2019','2020',
                    '2021','2022','2023','2024','2025'],
                   fontsize=10)
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax.tick_params(axis='both', labelsize=10)

# Легенда
ax.legend(loc='upper left', fontsize=10, framealpha=0.85,
          edgecolor='#E5E7EB')

# Значення в кінці ліній
for vals, color in zip([one_room, general, two_room, three_room],
                        ['#8B5CF6','#3B82F6','#10B981','#06B6D4']):
    ax.annotate(f'{vals[-1]:.0f}%',
                xy=(years[-1], vals[-1]),
                xytext=(5, 0), textcoords='offset points',
                fontsize=9, color=color, va='center')

plt.tight_layout()

OUT = os.path.join(os.path.dirname(__file__), 'plots',
                   '00_price_index_ukraine.png')
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f'Збережено: {OUT}')
