import { useState } from 'react'

const GROUP_NAMES = 'ABCDEFGHIJKL'.split('')

function advanceColor(pct) {
  if (pct >= 80) return '#3fb950'
  if (pct >= 60) return '#d29922'
  return '#8b949e'
}

function GroupTable({ groupName, group }) {
  const [showMatches, setShowMatches] = useState(false)

  return (
    <div className="group-card card">
      <div className="group-header">
        <span className="group-label">Group {groupName}</span>
        <button className="matches-toggle" onClick={() => setShowMatches(s => !s)}>
          {showMatches ? 'Hide matches ▲' : 'Show matches ▼'}
        </button>
      </div>

      <table className="sim-table">
        <thead>
          <tr>
            <th>Team</th>
            <th className="right">Avg Pts</th>
            <th className="right">GF</th>
            <th className="right">GA</th>
            <th style={{ minWidth: 130 }}>Advance%</th>
          </tr>
        </thead>
        <tbody>
          {group.teams.map((t, i) => (
            <tr key={t.name}>
              <td>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {i < 2 && <span className="advance-dot" style={{ background: advanceColor(t.advance_pct) }} />}
                  {i >= 2 && <span className="advance-dot" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }} />}
                  <span style={{ fontWeight: i < 2 ? 600 : 400 }}>{t.name}</span>
                  <span className="text-muted" style={{ fontSize: '0.72rem' }}>#{t.fifa_rank}</span>
                </div>
              </td>
              <td className="num">{t.avg_pts.toFixed(1)}</td>
              <td className="num">{t.avg_gf.toFixed(1)}</td>
              <td className="num">{t.avg_ga.toFixed(1)}</td>
              <td>
                <div className="prob-bar-wrap">
                  <div className="prob-bar-bg">
                    <div className="prob-bar-fill"
                      style={{ width: `${t.advance_pct}%`, background: advanceColor(t.advance_pct) }} />
                  </div>
                  <span className="prob-bar-label">{t.advance_pct.toFixed(0)}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {showMatches && (
        <div className="matches-table-wrap">
          <div style={{ height: '1px', background: 'var(--border)', margin: '0.75rem 0' }} />
          <table className="sim-table">
            <thead>
              <tr>
                <th>Home</th>
                <th>Away</th>
                <th className="right">Home W%</th>
                <th className="right">Draw%</th>
                <th className="right">Away W%</th>
                <th className="right">xGF</th>
                <th className="right">xGA</th>
              </tr>
            </thead>
            <tbody>
              {group.matches.map(m => (
                <tr key={`${m.home}-${m.away}`}>
                  <td style={{ fontWeight: 500 }}>{m.home}</td>
                  <td style={{ fontWeight: 500 }}>{m.away}</td>
                  <td className="num text-green">{m.home_win_pct.toFixed(1)}%</td>
                  <td className="num text-muted">{m.draw_pct.toFixed(1)}%</td>
                  <td className="num text-blue">{m.away_win_pct.toFixed(1)}%</td>
                  <td className="num">{m.xgf.toFixed(2)}</td>
                  <td className="num">{m.xga.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function GroupStage({ groups }) {
  return (
    <div>
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 className="section-title">Group Stage</h2>
        <p className="section-sub">
          Average points, goals, and advancement probability from {' '}
          <span style={{ color: 'var(--blue)' }}>50,000 simulations</span>.
          Top 2 plus 8 best third-place teams advance.
        </p>
      </div>
      <div className="groups-grid">
        {GROUP_NAMES.map(g => groups[g] && (
          <GroupTable key={g} groupName={g} group={groups[g]} />
        ))}
      </div>
      <style>{`
        .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.35rem; }
        .section-sub { color: var(--text-muted); font-size: 0.875rem; }
        .groups-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 1rem; }
        .group-card { padding: 1rem; }
        .group-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; }
        .group-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 700; }
        .matches-toggle {
          background: none; border: 1px solid var(--border); border-radius: 4px;
          color: var(--text-muted); cursor: pointer; font-size: 0.72rem; padding: 2px 8px;
        }
        .matches-toggle:hover { color: var(--text); border-color: var(--text-muted); }
        .advance-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .right { text-align: right; }
        .matches-table-wrap { overflow-x: auto; }
      `}</style>
    </div>
  )
}
