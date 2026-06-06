import { useState } from 'react'

const STAGES = [
  'Group Stage', 'Round of 32', 'Round of 16',
  'Quarterfinals', 'Semifinal', 'Final'
]

const STRATEGY_META = {
  EV_OPT:     { label: 'EV-Optimal', badge: 'badge-ev',     icon: '★', desc: 'Maximises log(survival ÷ pick%). Best strategy in a large pool.' },
  CHALK:      { label: 'Chalk',      badge: 'badge-chalk',  icon: '📈', desc: 'Pure survival probability. What ~70% of the pool does.' },
  CONTRARIAN: { label: 'Contrarian', badge: 'badge-contra', icon: '🎲', desc: 'Heaviest discount on over-picked teams. Highest variance.' },
}

function ProbBar({ value, max = 100, color = '#58a6ff' }) {
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-bg">
        <div className="prob-bar-fill" style={{ width: `${(value / max) * 100}%`, background: color }} />
      </div>
      <span className="prob-bar-label">{value.toFixed(1)}%</span>
    </div>
  )
}

function EVBadge({ ev }) {
  const color = ev >= 30 ? 'text-green' : ev >= 10 ? 'text-blue' : 'text-muted'
  return <span className={`${color}`} style={{ fontWeight: 600 }}>{ev.toFixed(1)}×</span>
}

function PickCard({ pick, index }) {
  return (
    <div className="pick-card">
      <div className="pick-number">#{index + 1}</div>
      <div className="pick-body">
        <div className="pick-team">
          <span className="pick-team-name">{pick.team}</span>
          <span className="pick-group">Group {pick.group}</span>
        </div>
        <div className="pick-metrics">
          <div className="pick-metric">
            <span className="metric-label">Survival</span>
            <ProbBar value={pick.survival_pct} color="#3fb950" />
          </div>
          <div className="pick-metric">
            <span className="metric-label">Est. Pick%</span>
            <ProbBar value={pick.pick_pct} max={30} color="#bc8cff" />
          </div>
          <div className="pick-metric-ev">
            <span className="metric-label">EV</span>
            <EVBadge ev={pick.ev_ratio} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default function SurvivorPicks({ survivor }) {
  const [strategy, setStrategy] = useState('EV_OPT')
  const picks = survivor.strategies[strategy]
  const meta  = STRATEGY_META[strategy]

  // Collect all teams used across all stages for this strategy
  const usedTeams = new Set()
  Object.values(picks).flat().forEach(p => usedTeams.add(p.team))

  return (
    <div className="survivor-view">
      <div className="survivor-header">
        <h2 className="section-title">Survivor Picks</h2>
        <p className="section-sub">
          Each team can only be used once. Picks are diversified across groups to reduce correlated failure.
        </p>
      </div>

      {/* Strategy selector */}
      <div className="strategy-tabs">
        {Object.entries(STRATEGY_META).map(([key, m]) => (
          <button
            key={key}
            className={`strategy-tab ${strategy === key ? 'active' : ''}`}
            onClick={() => setStrategy(key)}
          >
            <span className={`strategy-badge ${m.badge}`}>{m.icon} {m.label}</span>
          </button>
        ))}
      </div>
      <p className="strategy-desc">{meta.desc}</p>

      {/* All picks summary strip */}
      <div className="used-teams-strip card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title">All teams used in this strategy</div>
        <div className="used-teams-list">
          {[...usedTeams].map(t => (
            <span key={t} className="used-team-chip">{t}</span>
          ))}
        </div>
      </div>

      {/* Picks by stage */}
      <div className="stages-grid">
        {STAGES.map(stage => {
          const stagePicks = picks[stage]
          if (!stagePicks) return null
          return (
            <div key={stage} className="stage-section card">
              <div className="stage-header">
                <span className="card-title" style={{ margin: 0 }}>{stage}</span>
                <span className="stage-picks-count">
                  {stagePicks.length} pick{stagePicks.length > 1 ? 's' : ''}
                </span>
              </div>
              <div className="picks-list">
                {stagePicks.map((pick, i) => (
                  <PickCard key={pick.team} pick={pick} index={i} />
                ))}
              </div>
            </div>
          )
        })}
      </div>

      <style>{`
        .survivor-header { margin-bottom: 1.25rem; }
        .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.35rem; }
        .section-sub { color: var(--text-muted); font-size: 0.875rem; }

        .strategy-tabs { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
        .strategy-tab {
          background: var(--surface2); border: 1px solid var(--border);
          border-radius: var(--radius); cursor: pointer; padding: 0.5rem 1rem;
          transition: all 0.15s;
        }
        .strategy-tab.active { border-color: var(--blue); background: var(--blue-dim); }
        .strategy-tab:hover:not(.active) { border-color: var(--text-muted); }
        .strategy-desc { font-size: 0.82rem; color: var(--text-muted); margin-bottom: 1.25rem; }

        .used-teams-list { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .used-team-chip {
          background: var(--surface2); border: 1px solid var(--border);
          border-radius: 99px; font-size: 0.75rem; padding: 2px 10px;
          color: var(--text);
        }

        .stages-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1rem; }
        .stage-section { padding: 1rem 1.25rem; }
        .stage-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; }
        .stage-picks-count { font-size: 0.75rem; color: var(--text-muted); }
        .picks-list { display: flex; flex-direction: column; gap: 0.6rem; }

        .pick-card {
          display: flex; gap: 0.75rem; align-items: flex-start;
          background: var(--surface2); border-radius: 6px; padding: 0.65rem 0.75rem;
        }
        .pick-number { font-size: 0.75rem; font-weight: 700; color: var(--text-muted); padding-top: 2px; min-width: 1.5rem; }
        .pick-body { flex: 1; min-width: 0; }
        .pick-team { display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem; }
        .pick-team-name { font-weight: 600; font-size: 0.95rem; }
        .pick-group { font-size: 0.72rem; color: var(--text-muted); }
        .pick-metrics { display: flex; flex-direction: column; gap: 0.3rem; }
        .pick-metric { display: flex; align-items: center; gap: 0.5rem; }
        .pick-metric-ev { display: flex; align-items: center; gap: 0.5rem; }
        .metric-label { font-size: 0.72rem; color: var(--text-muted); width: 4.5rem; flex-shrink: 0; }
      `}</style>
    </div>
  )
}
