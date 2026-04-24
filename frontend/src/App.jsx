import { useState } from 'react'
import EstimateForm from './components/EstimateForm'
import ResultCard   from './components/ResultCard'

export default function App() {
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <>
      <header className="header">
        <div className="container">
          <h1>RealtyIQ</h1>
          <p>Інтелектуальна система оцінки вартості житлової нерухомості — Київ</p>
        </div>
      </header>

      <main className="container">
        <div className="card">
          <h2>Параметри об&apos;єкта</h2>
          <EstimateForm
            onResult={setResult}
            onError={setError}
            loading={loading}
            setLoading={setLoading}
          />
        </div>

        {error && (
          <div className="error-box">
            Помилка: {error}
          </div>
        )}

        {result && (
          <div className="card">
            <h2>Результат оцінки</h2>
            <ResultCard result={result} />
          </div>
        )}
      </main>
    </>
  )
}
