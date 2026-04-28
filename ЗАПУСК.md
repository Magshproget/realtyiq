# RealtyIQ — Інструкція із запуску

## Що потрібно

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — встановити і запустити

---

## Запуск

```bash
git clone https://github.com/Magshproget/realtyiq.git
cd realtyiq
docker compose up -d
```

Перший запуск займе ~3-5 хвилин (завантажує образи і встановлює залежності).  
Наступні запуски — ~20 секунд.

### Відкрити програму
Браузер: **http://localhost:5173**

### Зупинка
```bash
docker compose down
```

### Якщо змінили код — перебілдити
```bash
docker compose up -d --build
```

---

## Структура проекту

```
realtyiq/
├── docker-compose.yml
├── data/
│   └── class_flat_enriched.csv   ← дані квартир Києва (11 898 записів)
├── ml/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── predict.py                ← FastAPI ML-сервіс (порт 8000)
│   └── model/
│       ├── xgboost_model.pkl     ← навчена Stacking Optuna модель (R²=0.809)
│       └── interval_results.json
├── backend/
│   ├── Dockerfile
│   └── app.js                    ← Node.js REST API (порт 3001)
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── src/                      ← React SPA (порт 5173 → nginx)
```

---

## Тестові дані

| Поле           | Тест 1              | Тест 2              |
|----------------|---------------------|---------------------|
| Район          | Позняки             | Оболонь             |
| Кімнат         | 2                   | 1                   |
| Загальна площа | 55 м²               | 42 м²               |
| Житлова площа  | 32 м²               | 26 м²               |
| Площа кухні    | 9 м²                | 8 м²                |
| Поверх         | 4                   | 7                   |
| Поверховість   | 9                   | 16                  |
| Рік побудови   | 1990                | 2019                |
| Матеріал стін  | цегляний будинок    | монолітно-каркасний |
| Тип проекту    | хрущівка            | спец. проект        |
| ЖК             | ні                  | так                 |

---

## Якщо щось не запускається

### Порт 5173 зайнятий
```powershell
netstat -ano | findstr :5173
taskkill /F /PID <PID>
```

### Переглянути логи
```bash
docker compose logs -f ml       # ML сервіс
docker compose logs -f backend  # Node.js
docker compose logs -f frontend # nginx
```

### Docker не бачить зміни в коді
```bash
docker compose up -d --build
```
