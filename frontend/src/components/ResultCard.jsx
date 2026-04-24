export default function ResultCard({ result }) {
  const { predicted_price_sm, predicted_total,
          ci_lower_95, ci_upper_95, ci_lower_80, ci_upper_80,
          shap_values } = result

  const fmt  = n => n?.toLocaleString('uk-UA')
  const max  = ci_upper_95

  // ширина смуги відносно max
  const pct = (v) => `${Math.min(100, (v / max) * 100).toFixed(1)}%`

  return (
    <>
      {/* Головна ціна */}
      <div className="result-main">
        <div className="price-sm">{fmt(predicted_price_sm)} USD/м²</div>
        <div className="price-lbl">Прогнозована ціна за квадратний метр</div>
        <div className="price-total">Орієнтовна повна вартість: {fmt(predicted_total)} USD</div>
      </div>

      {/* Інтервали */}
      <div className="interval-block">
        <div className="interval-label">Довірчий інтервал прогнозу</div>

        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: '0.78rem', color: '#718096', marginBottom: 4 }}>95% інтервал</div>
          <div style={{ position: 'relative', height: 28, background: '#edf2f7', borderRadius: 6 }}>
            <div className="interval-bar bar-95"
              style={{ left: pct(ci_lower_95), width: `${((ci_upper_95 - ci_lower_95) / max * 100).toFixed(1)}%` }} />
          </div>
          <div className="interval-vals">
            <span>{fmt(ci_lower_95)} USD/м²</span>
            <span>{fmt(ci_upper_95)} USD/м²</span>
          </div>
        </div>

        <div>
          <div style={{ fontSize: '0.78rem', color: '#718096', marginBottom: 4 }}>80% інтервал</div>
          <div style={{ position: 'relative', height: 28, background: '#edf2f7', borderRadius: 6 }}>
            <div className="interval-bar bar-80"
              style={{ left: pct(ci_lower_80), width: `${((ci_upper_80 - ci_lower_80) / max * 100).toFixed(1)}%` }} />
          </div>
          <div className="interval-vals">
            <span>{fmt(ci_lower_80)} USD/м²</span>
            <span>{fmt(ci_upper_80)} USD/м²</span>
          </div>
        </div>
      </div>

      {/* SHAP */}
      {shap_values?.length > 0 && (
        <div>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#2d3748', marginBottom: 14 }}>
            Фактори впливу на ціну (SHAP-аналіз)
          </h3>
          {shap_values.slice(0, 8).map((sv, i) => {
            const isPos = sv.shap_value >= 0
            const maxShap = Math.max(...shap_values.map(s => Math.abs(s.shap_value)))
            const barW = `${(Math.abs(sv.shap_value) / maxShap * 100).toFixed(0)}%`
            return (
              <div key={i} className="shap-row">
                <div className="shap-name">{sv.feature}</div>
                <div className="shap-bar-wrap">
                  <div className={`shap-bar ${isPos ? 'shap-pos' : 'shap-neg'}`}
                    style={{ width: barW }}>
                    {isPos ? '+' : ''}{sv.shap_value.toFixed(3)}
                  </div>
                </div>
                <div className="shap-val">{isPos ? '▲' : '▼'}</div>
              </div>
            )
          })}
          <div style={{ fontSize: '0.75rem', color: '#a0aec0', marginTop: 8 }}>
            Зелений — підвищує ціну, червоний — знижує
          </div>
        </div>
      )}
    </>
  )
}
