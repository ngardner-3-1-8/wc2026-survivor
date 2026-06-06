import { useState } from 'react'
import { useSimData } from './useSimData'
import GroupStage from './components/GroupStage'
import KnockoutBracket from './components/KnockoutBracket'
import SurvivorPicks from './components/SurvivorPicks'
import PickIntelligence from './components/PickIntelligence'
import BracketView from './components/BracketView'
import BettingPage from './components/BettingPage'
import './app.css'

const TABS = [
  { id: 'survivor', label: '🎯 Survivor Picks' },
  { id: 'bracket',  label: '🗓 Bracket' },
  { id: 'betting',  label: '💰 Betting' },
  { id: 'groups',   label: '📊 Groups' },
  { id: 'knockout', label: '🏆 KO Odds' },
  { id: 'pickpct',  label: '🧠 Pick Intel' },
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
    </div>
  )

  const { meta, groups, knockout, survivor, bracket, betting } = data
  const updatedAt = new Date(meta.generated_at).toLocaleString('en-US', {
    month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', timeZoneName:'short'
  })

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-title">
            <span className="trophy">🏆</span>
            <div>
              <h1>WC 2026 Survivor League</h1>
              <p className="subtitle">Dixon-Coles Monte Carlo · {meta.n_sims.toLocaleString()} simulations</p>
            </div>
          </div>
          <div className="header-meta">Updated {updatedAt}</div>
        </div>
      </header>

      <nav className="tab-nav">
        {TABS.map(t => (
          <button key={t.id} className={`tab-btn ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {tab==='survivor' && <SurvivorPicks  survivor={survivor} />}
        {tab==='bracket'  && <BracketView    groups={groups} knockout={knockout} bracket={bracket} />}
        {tab==='betting'  && <BettingPage    betting={betting} />}
        {tab==='groups'   && <GroupStage     groups={groups} />}
        {tab==='knockout' && <KnockoutBracket knockout={knockout} />}
        {tab==='pickpct'  && <PickIntelligence survivor={survivor} />}
      </main>

      <footer className="app-footer">
        Model: Dixon-Coles Poisson · MLE-calibrated ratings · FIFA R32 bracket structure ·
        Pick% via softmax(survival · rank · media salience) · For entertainment purposes only.
      </footer>
    </div>
  )
}
