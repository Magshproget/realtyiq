const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

const app  = express();
const PORT = process.env.PORT || 3001;
const ML   = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'RealtyIQ Backend', port: PORT });
});

app.post('/api/estimates', async (req, res) => {
  console.log('POST /api/estimates body:', JSON.stringify(req.body).slice(0, 80));
  const { hist_district, rooms, area_total, area_living,
          area_kitchen, stock, stock_total, year,
          wall, project, is_complex, lat, lon } = req.body;

  const errors = [];
  if (!hist_district)                 errors.push('hist_district обовязковий');
  if (!rooms || rooms < 1)            errors.push('rooms >= 1');
  if (!area_total || area_total < 15) errors.push('area_total >= 15');
  if (!stock || stock < 1)            errors.push('stock >= 1');
  if (!stock_total)                   errors.push('stock_total обовязковий');
  if (!year || year < 1900)           errors.push('year >= 1900');
  if (errors.length) return res.status(400).json({ success: false, errors });

  try {
    const { data: ml } = await axios.post(`${ML}/predict`, {
      hist_district,
      rooms:        Number(rooms),
      area_total:   Number(area_total),
      area_living:  area_living  ? Number(area_living)  : Number(area_total) * 0.6,
      area_kitchen: area_kitchen ? Number(area_kitchen) : 10,
      stock:        Number(stock),
      stock_total:  Number(stock_total),
      year:         Number(year),
      wall:         wall    || 'Невідомо',
      project:      project || 'спец. проект',
      is_complex:   is_complex ? 1 : 0,
      ...(lat != null && lon != null ? { lat: Number(lat), lon: Number(lon) } : {}),
    });

    return res.json({
      success:            true,
      estimate_id:        null,
      predicted_price_sm: Math.round(ml.predicted_price_sm),
      predicted_total:    Math.round(ml.predicted_price_sm * Number(area_total)),
      ci_lower_95:        Math.round(ml.ci_lower_95),
      ci_upper_95:        Math.round(ml.ci_upper_95),
      ci_lower_80:        Math.round(ml.ci_lower_80),
      ci_upper_80:        Math.round(ml.ci_upper_80),
      shap_values:        ml.shap_values || [],
    });
  } catch (err) {
    console.error('ML error:', err.code, err.message);
    return res.status(503).json({ success: false, error: 'ML сервiс недоступний: ' + err.message });
  }
});

app.get('/api/estimates', (req, res) => res.json({ success: true, data: [] }));

app.listen(PORT, () => console.log(`RealtyIQ Backend: http://localhost:${PORT}`));
module.exports = app;
