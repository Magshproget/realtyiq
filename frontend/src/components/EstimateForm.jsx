import { useState } from 'react'
import MapPicker from './MapPicker'

const DISTRICTS = [
  'Позняки','Нова забудова','Оболонь','Чорна Гора',
  'Нижній Печерськ','Троєщина','Осокорки','Виноградар',
  'Печерськ','Микільська Слобідка','Лівобережний масив',
  'Шулявка','Солом\'янка','Голосіїво','Теремки-2',
  'Сирець','Борщагівка','Деміївка','Лук\'янівка',
  'Нова Дарниця','Харьківський','Липки','Нивки',
  'Звіринець','Історичний центр','Поділ','Святошино',
  'Теремки','Русанівка','Куренівка','Невідомо'
]
const WALLS = [
  'монолітно-каркасний',
  'цегляний будинок',
  'утеплена панель',
  'панельні',
  'блочні',
  'Невідомо',
]
const PROJECTS = [
  'спец. проект',
  'хрущівка',
  'сталінка',
  'серія 96',
  'серія КТ',
  'чеський проект',
  'АППС',
  'Невідомо',
]

export default function EstimateForm({ onResult, onError, loading, setLoading }) {
  const [form, setForm] = useState({
    hist_district: 'Позняки',
    rooms:         2,
    area_total:    55,
    area_living:   32,
    area_kitchen:  9,
    stock:         4,
    stock_total:   9,
    year:          1990,
    wall:          'монолітно-каркасний',
    project:       'спец. проект',
    is_complex:    false,
  })
  const [lat,     setLat]     = useState(null)
  const [lon,     setLon]     = useState(null)
  const [showMap, setShowMap] = useState(false)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    onError(null)
    try {
      const body = {
        ...form,
        rooms:        Number(form.rooms),
        area_total:   Number(form.area_total),
        area_living:  Number(form.area_living),
        area_kitchen: Number(form.area_kitchen),
        stock:        Number(form.stock),
        stock_total:  Number(form.stock_total),
        year:         Number(form.year),
      }
      if (lat !== null && lon !== null) {
        body.lat = lat
        body.lon = lon
      }

      const res  = await fetch('/api/estimates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!data.success) throw new Error(data.errors?.join(', ') || data.error)
      onResult(data)
    } catch (err) {
      onError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const clearLocation = () => { setLat(null); setLon(null) }

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-grid">
        <div className="form-group">
          <label>Район міста</label>
          <select value={form.hist_district} onChange={e => set('hist_district', e.target.value)}>
            {DISTRICTS.map(d => <option key={d}>{d}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Кількість кімнат</label>
          <input type="number" min="1" max="6" value={form.rooms}
            onChange={e => set('rooms', e.target.value)} required />
        </div>

        <div className="form-group">
          <label>Загальна площа, м²</label>
          <input type="number" min="15" max="250" step="0.5" value={form.area_total}
            onChange={e => set('area_total', e.target.value)} required />
        </div>

        <div className="form-group">
          <label>Житлова площа, м²</label>
          <input type="number" min="10" step="0.5" value={form.area_living}
            onChange={e => set('area_living', e.target.value)} />
        </div>

        <div className="form-group">
          <label>Площа кухні, м²</label>
          <input type="number" min="4" step="0.5" value={form.area_kitchen}
            onChange={e => set('area_kitchen', e.target.value)} />
        </div>

        <div className="form-group">
          <label>Поверх</label>
          <input type="number" min="1" value={form.stock}
            onChange={e => set('stock', e.target.value)} required />
        </div>

        <div className="form-group">
          <label>Поверховість будинку</label>
          <input type="number" min="1" value={form.stock_total}
            onChange={e => set('stock_total', e.target.value)} required />
        </div>

        <div className="form-group">
          <label>Рік побудови</label>
          <input type="number" min="1900" max="2024" value={form.year}
            onChange={e => set('year', e.target.value)} required />
        </div>

        <div className="form-group">
          <label>Матеріал стін</label>
          <select value={form.wall} onChange={e => set('wall', e.target.value)}>
            {WALLS.map(w => <option key={w}>{w}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Тип проекту</label>
          <select value={form.project} onChange={e => set('project', e.target.value)}>
            {PROJECTS.map(p => <option key={p}>{p}</option>)}
          </select>
        </div>
      </div>

      <div className="form-group" style={{ marginTop: 14 }}>
        <label style={{ flexDirection: 'row', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input type="checkbox" checked={form.is_complex}
            onChange={e => set('is_complex', e.target.checked)} />
          &nbsp; Квартира в житловому комплексі (ЖК)
        </label>
      </div>

      {/* Блок вибору локації */}
      <div className="location-section">
        <div className="location-header">
          <div>
            <div className="location-title">Місцезнаходження на карті</div>
            <div className="location-hint">
              {lat && lon
                ? `Вказано: ${lat.toFixed(5)}, ${lon.toFixed(5)}`
                : 'Необов\'язково — підвищує точність прогнозу'}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {lat && lon && (
              <button type="button" className="btn-clear-loc" onClick={clearLocation}>
                Скинути
              </button>
            )}
            <button type="button" className="btn-map-toggle"
              onClick={() => setShowMap(v => !v)}>
              {showMap ? 'Сховати карту' : (lat ? 'Змінити на карті' : 'Вибрати на карті')}
            </button>
          </div>
        </div>

        {lat && lon && !showMap && (
          <div className="location-badge">
            Координати: {lat.toFixed(5)}, {lon.toFixed(5)}
          </div>
        )}

        {showMap && (
          <div className="map-wrapper">
            <MapPicker lat={lat} lon={lon}
              onChange={(la, lo) => { setLat(la); setLon(lo) }} />
            <div className="map-hint">
              Клікніть на карті, щоб вказати точне місцезнаходження квартири
            </div>
          </div>
        )}
      </div>

      <button type="submit" className="btn-submit" disabled={loading}>
        {loading ? <><span className="spinner" />Розраховую...</> : 'Розрахувати вартість'}
      </button>
    </form>
  )
}
