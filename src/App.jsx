import { useMemo, useState } from 'react'
import './App.css'

const mockEmotionResult = {
  label: 'Calm',
  confidence: 0.86,
  cues: ['Soft wording', 'Neutral sentiment', 'Steady pacing'],
}

function App() {
  const [mode, setMode] = useState('text')
  const [textInput, setTextInput] = useState('')
  const [gender, setGender] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)

  const isReadyToAnalyze = useMemo(() => {
    if (mode === 'text') return textInput.trim().length > 0 && gender
    return false
  }, [mode, textInput])

  const handleAnalyze = () => {
    if (!isReadyToAnalyze) return
    setIsAnalyzing(true)
    // Placeholder for future model call
    setTimeout(() => {
      setResult(mockEmotionResult)
      setIsAnalyzing(false)
    }, 650)
  }

  const handleAudioPlaceholder = () => {
    alert('Audio analysis will be available once the model is integrated.')
  }

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Truetone AI</p>
          <h1>Express yourself through text or voice</h1>
          <p className="sub">
            Share your thoughts to detect emotion.
          </p>
        </div>
      </header>

      <section className="panel highlight">
        <div className="panel-body">
          <div className="panel-graphic">
            <div className="badge">Live soon</div>
            <div className="masks">
              <span aria-hidden>🎭</span>
            </div>
            <p>Type or speak to detect emotion</p>
          </div>
          <div className="panel-card">
            <div className="tabs">
              <button
                className={mode === 'text' ? 'tab active' : 'tab'}
                onClick={() => setMode('text')}
              >
                Text
              </button>
              <button
                className={mode === 'speech' ? 'tab active' : 'tab'}
                onClick={() => setMode('speech')}
              >
                Speech
              </button>
            </div>

            {mode === 'text' ? (
              <label className="input-stack">
                <span className="label">Input text to analyze?</span>
                <textarea
                  placeholder="Input a thought or short message..."
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  rows={6}
                />
              </label>
            ) : (
              <div className="input-stack">
                <span className="label">Upload or record speech</span>
                <div className="upload-tile" onClick={handleAudioPlaceholder}>
                  <div className="upload-icon">🎤</div>
                  <div>
                    <p className="upload-title">Voice upload coming soon</p>
                    <p className="upload-sub">
                      We&apos;ll enable audio upload and live recording after
                      the model is wired up.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="choice-group">
              <span className="label">Speaker gender</span>
              <div className="segmented">
                {[
                  { value: 'female', label: 'Female' },
                  { value: 'male', label: 'Male' },
                ].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={
                      gender === option.value ? 'chip active' : 'chip'
                    }
                    onClick={() => setGender(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              className="analyze"
              disabled={!isReadyToAnalyze || isAnalyzing}
              onClick={handleAnalyze}
            >
              {isAnalyzing ? 'Analyzing…' : 'Analyze'}
            </button>

            <div className="divider" />

            <div className="result">
              <div className="result-header">
                <span className="result-title">Emotion</span>
                <span className="pill">Model pending</span>
              </div>
              {result ? (
                <>
                  <p className="emotion-label">{result.label}</p>
                  <p className="confidence">
                    Confidence: {(result.confidence * 100).toFixed(0)}%
                  </p>
                  <ul className="cues">
                    {result.cues.map((cue) => (
                      <li key={cue}>{cue}</li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="placeholder">
                  Results will appear here once the model is connected.
                </p>
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
