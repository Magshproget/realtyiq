const axios = require('axios');
require('dotenv').config();

const ML_URL = 'http://127.0.0.1:8000';

// POST /api/estimates
const createEstimate = async (req, res) => {
  console.log('>>> createEstimate called, ML_URL:', ML_URL);
  try {
    const {
      hist_district, rooms, area_total, area_living,
      area_kitchen, stock, stock_total, year,
      wall, project, is_complex,
    } = req.body;

    // 1. Валідація
    const errors = [];
    if (!hist_district)                 errors.push('hist_district обовязковий');
    if (!rooms || rooms < 1)            errors.push('rooms >= 1');
    if (!area_total || area_total < 15) errors.push('area_total >= 15');
    if (!stock || stock < 1)            errors.push('stock >= 1');
    if (!stock_total)                   errors.push('stock_total обовязковий');
    if (!year || year < 1900)           errors.push('year >= 1900');
    if (errors.length > 0) {
      return res.status(400).json({ success: false, errors });
    }

    // 2. Запит до ML-сервісу
    let mlData;
    try {
      const mlResponse = await axios.post(`${ML_URL}/predict`, {
        hist_district,
        rooms:        Number(rooms),
        area_total:   Number(area_total),
        area_living:  area_living  ? Number(area_living)  : Number(area_total) * 0.6,
        area_kitchen: area_kitchen ? Number(area_kitchen) : 10,
        stock:        Number(stock),
        stock_total:  Number(stock_total),
        year:         Number(year),
        wall:         wall    || 'Невідомо',
        project:      project || 'Невідомо',
        is_complex:   is_complex ? 1 : 0,
      });
      mlData = mlResponse.data;
    } catch (mlErr) {
      console.error('ML service error:', mlErr.code, mlErr.message, 'URL:', ML_URL);
      return res.status(503).json({
        success: false,
        error: 'ML-сервiс недоступний. Перевiрте що Python FastAPI запущений на порту 8000.',
      });
    }

    // 3. Відповідь (без БД — PostgreSQL опціональний)
    return res.status(200).json({
      success:            true,
      estimate_id:        null,
      predicted_price_sm: Math.round(mlData.predicted_price_sm),
      predicted_total:    Math.round(mlData.predicted_price_sm * Number(area_total)),
      ci_lower_95:        Math.round(mlData.ci_lower_95),
      ci_upper_95:        Math.round(mlData.ci_upper_95),
      ci_lower_80:        Math.round(mlData.ci_lower_80),
      ci_upper_80:        Math.round(mlData.ci_upper_80),
      shap_values:        mlData.shap_values || [],
    });

  } catch (err) {
    console.error('estimateController error:', err.message);
    return res.status(500).json({ success: false, error: err.message });
  }
};

// GET /api/estimates
const getHistory = async (req, res) => {
  return res.status(200).json({ success: true, data: [] });
};

// GET /api/estimates/:id
const getEstimateById = async (req, res) => {
  return res.status(404).json({ success: false, error: 'Not found' });
};

module.exports = { createEstimate, getHistory, getEstimateById };
