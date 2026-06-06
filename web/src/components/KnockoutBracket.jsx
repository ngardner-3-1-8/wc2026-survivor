import { useState, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const STAGES = [
  { key: 'r32_pct',      label: 'R32' },
  { key: 'r16_pct',      label: 'R16' },
  { key: 'qf_pct',       label: 'QF' },
  { key: 'sf_pct',       label: 'SF' },
  { key: 'final_pct',    label: 'Final' },
  { key: 'champion_pct', label: '🏆 Win' },
]

const STAGE_COLORS = {
  r32_pct:      '#58a6ff',
  r16_pct:      '#79c0ff',
  qf_pct:       '#bc8cff',
  sf_pct:       '#d2a8ff',
  final_pct:    '#f78166',
  champion_pct: '#ffd700',
}

function pctColor(val, key) {
  if (key === 'champion_pct') {
    if (val >= 10) return 'text-yellow'
    if (val >= 5)  return 'text-blue'
    return 'text-muted'
  }
  if (val >= 70) return 'text-green'
  if (val >= 40) return 'text-blue'
  if (val >= 15) return ''
  return 'text-muted'
}

export default function KnockoutBracket({ knockout }) {
  const [sortKey, setSortKey] = useState('champion_pct')
  const [filterGroup, setFilterGroup] = useState('ALL')
  const [showChart, setShowChart] = useState(false)

  const groups = ['ALL', ...new Set(knockout.map(t => t.group)).values()].sort()

  const sorted = useMemo(() => {
    let teams = [...knockout]
    if (filterGroup !== 'ALL') teams = teams.filter(t => t.group === filterGroup)
    return teams.sort((a, b) => b[sortKey] - a[sortKey])
  }, [knockout, sortKey, filterGroup])

  const chartData = useMemo(() =>
    sorted.slice(0, 16).map(t => ({
      name: t.name,
      ...Object.fromEntries(STAGES.map(s => [s.label, t[s.key]]))
    }))
  , [sorted])

  return (
    <div>
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 className="section-title">Knockout Probabilities</h2>
        <p className="section-sub">
          Probability of each team reaching each knockout round, across all simulations.
        </p>
      </div>

      <div className="ko-controls card" style={{ marginBottom: '1rem', display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <div>
          <label className="ctrl-label">Sort by</label>
          <div className="btn-group">
            {STAGES.map(s => (
              <button
                key={s.key}
                className={`ctrl-btn ${sortKey === s.key ? 'active' : ''}`}
                onClick={() => setSortKey(s.key)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="ctrl-label">Group</label>
          <div className="btn-group">
            {groups.map(g => (
              <button
                key={g}
                className={`ctrl-btn ${filterGroup === g ? 'active' : ''}`}
                onClick={() => setFilterGroup(g)}
              >
                {g}
              </button>
            ))}
          </div>
        </div>
        <button
          className={`ctrl-btn ${showChart ? 'active' : ''}`}
          onClick={() => setShowChart(s => !s)}
          style={{ marginLeft: 'auto' }}
        >
          {showChart ? '📋 Table' : '📊 Chart'}
        </button>
      </div>

      {showChart ? (
        <div className="card" style={{ padding: '1rem', height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
              <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 11 }} angle={-40} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 6 }}
                labelStyle={{ color: '#e6edf3', fontWeight: 600 }}
                formatter={(val) => [`${val.toFixed(1)}%`]}
              />
              {STAGES.map(s => (
                <Bar key={s.key} dataKey={s.label} stackId="a" fill={STAGE_COLORS[s.key]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="sim-table">
              <thead>
                <tr>
                  <th style={{ width: 24 }}>#</th>
                  <th>Team</th>
                  <th>Grp</th>
                  {STAGES.map(s => (
                    <th key={s.key}
                      className="num"
                      style={{ cursor: 'pointer', color: sortKey === s.key ? 'var(--blue)' : undefined }}
                      onClick={() => setSortKey(s.key)}
                    >
                      {s.label} {sortKey === s.key ? '▼' : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((t, i) => (
                  <tr key={t.name}>
                    <td className="text-muted" style={{ fontSize: '0.78rem' }}>{i + 1}</td>
                    <td style={{ fontWeight: 500 }}>{t.name}</td>
                    <td className="text-muted">{t.group}</td>
                    {STAGES.map(s => (
                      <td key={s.key} className={`num ${pctColor(t[s.key], s.key)}`}>
                        {t[s.key].toFixed(1)}%
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <style>{`
        .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.35rem; }
        .section-sub { color: var(--text-muted); font-size: 0.875rem; }
        .ko-controls { padding: 0.75rem 1rem; }
        .ctrl-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.35rem; }
        .btn-group { display: flex; flex-wrap: wrap; gap: 0.25rem; }
        .ctrl-btn {
          background: var(--surface2); border: 1px solid var(--border);
          border-radius: 4px; color: var(--text-muted); cursor: pointer;
          font-size: 0.78rem; padding: 3px 8px; transition: all 0.12s;
        }
        .ctrl-btn:hover  { color: var(--text); border-color: var(--text-muted); }
        .ctrl-btn.active { background: var(--blue-dim); border-color: var(--blue); color: var(--blue); }
      `}</style>
    </div>
  )
}
