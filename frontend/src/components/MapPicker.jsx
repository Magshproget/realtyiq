import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix default marker icons broken by Vite's asset bundling
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon   from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl:       markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl:     markerShadow,
})

export default function MapPicker({ lat, lon, onChange }) {
  const containerRef = useRef(null)
  const mapRef       = useRef(null)
  const markerRef    = useRef(null)

  useEffect(() => {
    if (mapRef.current) return

    const map = L.map(containerRef.current, { zoomControl: true })
      .setView([50.4501, 30.5234], 12)

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(map)

    map.on('click', (e) => {
      const { lat: clickLat, lng: clickLon } = e.latlng
      placeMarker(map, clickLat, clickLon)
      onChange(+clickLat.toFixed(6), +clickLon.toFixed(6))
    })

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Restore marker if lat/lon already set (e.g. on re-show)
  useEffect(() => {
    if (!mapRef.current || !lat || !lon) return
    placeMarker(mapRef.current, lat, lon)
    mapRef.current.setView([lat, lon], 14)
  }, [])

  function placeMarker(map, la, lo) {
    if (markerRef.current) markerRef.current.remove()
    markerRef.current = L.marker([la, lo])
      .addTo(map)
      .bindPopup(`<b>${la.toFixed(5)}, ${lo.toFixed(5)}</b><br>Клікніть ще раз, щоб змінити`)
      .openPopup()
  }

  return (
    <div ref={containerRef} className="map-container" />
  )
}
