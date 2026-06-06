import { useState } from 'react'
import { useSimData } from './useSimData'
import GroupStage from './components/GroupStage'
import KnockoutBracket from './components/KnockoutBracket'
import SurvivorPicks from './components/SurvivorPicks'
import PickIntelligence from './components/PickIntelligence'
import BracketView from './components/BracketView'
import './app.css'

const TABS = [
  { id: 'survivor',  label: '🎯 Survivor Picks' },
  { id: 'bracket',   label: '🗓 Bracket' },
  { id: 'groups',    label: '📊 Group Stage' },
  { id: 'knockout',  label: '🏆 Knockout Odds' },
  { id: 'pickpct',   label: '🧠 Pick Intelligence' },
]

export default function App() {
  const { data, error, loading } = useSimData()
  const [tab, setTab] = useState('survivor')

  if (loading) return (
    <div className="loading">
      <div className="loading-ball">⚽</div>
      <p>Running simulations…</p>
    </div>
  )
  if (error) return (
    <div className="error">
      <h2>Failed to load simulation data</h2>
      <pre>{error.message}</pre>
      <p>Make sure you've run <code>python export_json.py</code> and copied the JSON to <code>public/data/</code>.</p>
    </div>
  )

  const { meta, groups, knockout, survivor } = data
  const updatedAt = new Date(meta.generated_at).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short'
  })

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-title">
            <span className="trophy">🏆</span>
            <div>
              <h1>WC 2026 Survivor League</h1>
              <p className="subtitle">Monte Carlo simulator · Dixon-Coles model · {meta.n_sims.toLocaleString()} simulations</p>
            </div>
          </div>
          <div className="header-meta">
            Updated {updatedAt}
          </div>
        </div>
      </header>

      <nav className="tab-nav">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {tab === 'survivor'  && <SurvivorPicks  survivor={survivor} />}
        {tab === 'bracket'   && <BracketView    groups={groups} knockout={knockout} />}
        {tab === 'groups'    && <GroupStage     groups={groups} />}
        {tab === 'knockout'  && <KnockoutBracket knockout={knockout} />}
        {tab === 'pickpct'   && <PickIntelligence survivor={survivor} />}
      </main>

      <footer className="app-footer">
        Model: Dixon-Coles Poisson process with MLE-calibrated ratings &amp; confederation SOS adjustment.
        Pick% estimated via softmax(survival · rank · media salience). For entertainment purposes.
      </footer>
    </div>
  )
}
