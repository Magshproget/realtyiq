const express  = require('express');
const router   = express.Router();
const ctrl     = require('../controllers/estimateController');

router.post  ('/',     ctrl.createEstimate);
router.get   ('/',     ctrl.getHistory);
router.get   ('/:id',  ctrl.getEstimateById);

module.exports = router;
