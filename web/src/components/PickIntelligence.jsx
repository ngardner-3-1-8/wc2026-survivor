import { useState } from 'react'

const STAGES = [
  'Group Stage', 'Round of 32', 'Round of 16',
  'Quarterfinals', 'Semifinal', 'Final'
]

function ValueChip({ label }) {
  if (label.includes('GREAT')) return <span className="value-label value-great">★ Great Value</span>
  if (label.includes('AVOID')) return <span className="value-label value-avoid">↓ Avoid</span>
  return <span className="value-label value-ok">OK</span>
}

function ScatterDot({ surv, pick, team, isSelected }) {
  // Map surv (0-100) and pick (0-30) to SVG coords
  const x = (surv / 100) * 340 + 40
  const y = 220 - (Math.min(pick, 25) / 25) * 200
  return (
    <g>
      <circle
        cx={x} cy={y} r={isSelected ? 7 : 5}
        fill={isSelected ? '#ffd700' : '#58a6ff'}
        fillOpacity={isSelected ? 1 : 0.6}
        stroke={isSelected ? '#ffd700' : 'none'}
        strokeWidth={2}
      />
      {isSelected && (
        <text x={x + 9} y={y + 4} fill="#e6edf3" fontSize="10" fontWeight="600">{team}</text>
      )}
    </g>
  )
}

export default function PickIntelligence({ survivor }) {
  const [stage, setStage] = useState('Group Stage')
  const [highlight, setHighlight] = useState(null)
  const intel = survivor.pick_intelligence[stage] || []

  return (
    <div>
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 className="section-title">Pick Intelligence</h2>
        <p className="section-sub">
          Estimated public pick% vs survival probability per stage. Teams in the top-left are 
          {' '}<strong style={{ color: 'var(--green)' }}>undervalued by the field</strong> — high survival, low pick%.
          Teams in the bottom-right are{' '}<strong style={{ color: 'var(--red)' }}>over-picked</strong>.
        </p>
      </div>

      {/* Stage tabs */}
      <div className="stage-tabs" style={{ marginBottom: '1rem' }}>
        {STAGES.filter(s => survivor.pick_intelligence[s]).map(s => (
          <button
            key={s}
            className={`ctrl-btn ${stage === s ? 'active' : ''}`}
            onClick={() => setStage(s)}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="intel-layout">
        {/* Scatter plot */}
        <div className="card scatter-card">
          <div className="card-title">Survival% vs Estimated Pick%</div>
          <svg viewBox="0 0 420 260" style={{ width: '100%', overflow: 'visible' }}>
            {/* Axes */}
            <line x1="40" y1="220" x2="380" y2="220" stroke="#30363d" strokeWidth="1" />
            <line x1="40" y1="20"  x2="40"  y2="220" stroke="#30363d" strokeWidth="1" />
            {/* Axis labels */}
            <text x="210" y="255" fill="#8b949e" fontSize="10" textAnchor="middle">Survival %</text>
            <text x="10" y="120" fill="#8b949e" fontSize="10" textAnchor="middle" transform="rotate(-90 10 120)">Pick %</text>
            {/* X ticks */}
            {[0, 25, 50, 75, 100].map(v => (
              <g key={v}>
                <line x1={40 + v*3.4} y1="220" x2={40 + v*3.4} y2="225" stroke="#30363d" />
                <text x={40 + v*3.4} y="235" fill="#8b949e" fontSize="9" textAnchor="middle">{v}%</text>
              </g>
            ))}
            {/* Y ticks */}
            {[0, 5, 10, 15, 20, 25].map(v => (
              <g key={v}>
                <line x1="35" y1={220 - (v/25)*200} x2="40" y2={220 - (v/25)*200} stroke="#30363d" />
                <text x="30" y={224 - (v/25)*200} fill="#8b949e" fontSize="9" textAnchor="end">{v}%</text>
              </g>
            ))}
            {/* "Good value" region annotation */}
            <rect x="240" y="140" width="130" height="75" fill="#3fb950" fillOpacity="0.05" stroke="#3fb950" strokeOpacity="0.2" strokeDasharray="3,3" rx="4" />
            <text x="305" y="158" fill="#3fb950" fontSize="9" textAnchor="middle" opacity="0.6">HIGH SURV / LOW PICK%</text>
            <text x="305" y="170" fill="#3fb950" fontSize="9" textAnchor="middle" opacity="0.6">(target zone)</text>
            {/* Dots */}
            {intel.map(t => (
              <ScatterDot
                key={t.team}
                surv={t.survival_pct}
                pick={t.pick_pct}
                team={t.team}
                isSelected={highlight === t.team}
              />
            ))}
          </svg>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Hover a row in the table to highlight team. Target zone = high survival + low public pick%.
          </p>
        </div>

        {/* Table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden', flex: 1 }}>
          <div style={{ overflowY: 'auto', maxHeight: 420 }}>
            <table className="sim-table">
              <thead style={{ position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1 }}>
                <tr>
                  <th>Team</th>
                  <th>Grp</th>
                  <th className="num">Surv%</th>
                  <th className="num">Pick%</th>
                  <th className="num">EV</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {intel.map(t => (
                  <tr
                    key={t.team}
                    onMouseEnter={() => setHighlight(t.team)}
                    onMouseLeave={() => setHighlight(null)}
                    style={{ cursor: 'default', background: highlight === t.team ? 'var(--surface2)' : undefined }}
                  >
                    <td style={{ fontWeight: 500 }}>{t.team}</td>
                    <td className="text-muted">{t.group}</td>
                    <td className="num">{t.survival_pct.toFixed(1)}%</td>
                    <td className="num text-purple">{t.pick_pct.toFixed(1)}%</td>
                    <td className="num">
                      <span style={{
                        fontWeight: 600,
                        color: t.ev_ratio >= 30 ? 'var(--green)' : t.ev_ratio >= 10 ? 'var(--blue)' : 'var(--text-muted)'
                      }}>
                        {t.ev_ratio.toFixed(1)}×
                      </span>
                    </td>
                    <td><ValueChip label={t.value_label} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card ev-explainer" style={{ marginTop: '1rem' }}>
        <div className="card-title">How to read EV ratio</div>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          <strong style={{ color: 'var(--text)' }}>EV ratio = survival% ÷ pick%</strong>. 
          A ratio of <strong style={{ color: 'var(--green)' }}>50×</strong> means if the team survives, 
          you beat roughly 98% of the entries who didn't pick them. 
          A ratio of <strong style={{ color: 'var(--text-muted)' }}>5×</strong> means you only beat 80% — 
          because 20% of the field also picked that team and also survived.
          In a 20,000-entry pool, <em>differentiation is how you win</em>, not just survival.
        </p>
      </div>

      <style>{`
        .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.35rem; }
        .section-sub { color: var(--text-muted); font-size: 0.875rem; }
        .stage-tabs { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .ctrl-btn {
          background: var(--surface2); border: 1px solid var(--border);
          border-radius: 4px; color: var(--text-muted); cursor: pointer;
          font-size: 0.78rem; padding: 3px 10px; transition: all 0.12s;
        }
        .ctrl-btn:hover  { color: var(--text); }
        .ctrl-btn.active { background: var(--blue-dim); border-color: var(--blue); color: var(--blue); }
        .intel-layout { display: flex; gap: 1rem; flex-wrap: wrap; }
        .scatter-card { flex: 0 0 420px; max-width: 100%; }
        .text-purple { color: var(--purple); }
        .ev-explainer { padding: 0.85rem 1.1rem; }
        .num { text-align: right; }
      `}</style>
    </div>
  )
}
