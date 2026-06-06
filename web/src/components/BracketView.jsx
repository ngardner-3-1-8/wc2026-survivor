import { useState, useMemo } from 'react'

// Real FIFA R32 matchup structure (Match 73-88)
// Each matchup: { id, home: {slot, label}, away: {slot, label} }
const R32_STRUCTURE = [
  { id: 'M73', home: { slot: '2A', label: 'Runner-up A' }, away: { slot: '2B', label: 'Runner-up B' } },
  { id: 'M74', home: { slot: '1E', label: 'Winner E'    }, away: { slot: '3rd', label: 'Best 3rd (A/B/C/D/F)' } },
  { id: 'M75', home: { slot: '1F', label: 'Winner F'    }, away: { slot: '2C', label: 'Runner-up C' } },
  { id: 'M76', home: { slot: '1C', label: 'Winner C'    }, away: { slot: '2F', label: 'Runner-up F' } },
  { id: 'M77', home: { slot: '1I', label: 'Winner I'    }, away: { slot: '3rd', label: 'Best 3rd (C/D/F/G/H)' } },
  { id: 'M78', home: { slot: '2E', label: 'Runner-up E' }, away: { slot: '2I', label: 'Runner-up I' } },
  { id: 'M79', home: { slot: '1A', label: 'Winner A'    }, away: { slot: '3rd', label: 'Best 3rd (C/E/F/H/I)' } },
  { id: 'M80', home: { slot: '1L', label: 'Winner L'    }, away: { slot: '3rd', label: 'Best 3rd (E/H/I/J/K)' } },
  { id: 'M81', home: { slot: '1D', label: 'Winner D'    }, away: { slot: '3rd', label: 'Best 3rd (B/E/F/I/J)' } },
  { id: 'M82', home: { slot: '1G', label: 'Winner G'    }, away: { slot: '3rd', label: 'Best 3rd (A/E/H/I/J)' } },
  { id: 'M83', home: { slot: '2K', label: 'Runner-up K' }, away: { slot: '2L', label: 'Runner-up L' } },
  { id: 'M84', home: { slot: '1H', label: 'Winner H'    }, away: { slot: '2J', label: 'Runner-up J' } },
  { id: 'M85', home: { slot: '1B', label: 'Winner B'    }, away: { slot: '3rd', label: 'Best 3rd (E/F/G/I/J)' } },
  { id: 'M86', home: { slot: '1J', label: 'Winner J'    }, away: { slot: '2H', label: 'Runner-up H' } },
  { id: 'M87', home: { slot: '1K', label: 'Winner K'    }, away: { slot: '3rd', label: 'Best 3rd (D/E/I/J/L)' } },
  { id: 'M88', home: { slot: '2D', label: 'Runner-up D' }, away: { slot: '2G', label: 'Runner-up G' } },
]

// Bracket halves: M73-80 feed into one half, M81-88 the other
const BRACKET_LEFT  = ['M73','M74','M75','M76','M77','M78','M79','M80']
const BRACKET_RIGHT = ['M81','M82','M83','M84','M85','M86','M87','M88']

function pctBar(pct, color = '#58a6ff') {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 }}>
      <div style={{ flex: 1, height: 4, background: '#21262d', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: 10, color: '#8b949e', width: 32, textAlign: 'right' }}>{pct.toFixed(1)}%</span>
    </div>
  )
}

function TeamSlot({ name, group, pct, stage, highlight, onHover }) {
  const ko = { r32: 'r32_pct', r16: 'r16_pct', qf: 'qf_pct', sf: 'sf_pct', final: 'final_pct', champion: 'champion_pct' }
  const color = pct > 60 ? '#3fb950' : pct > 35 ? '#58a6ff' : pct > 15 ? '#d29922' : '#8b949e'
  const isHL = highlight === name

  return (
    <div
      className={`team-slot ${isHL ? 'highlighted' : ''}`}
      onMouseEnter={() => onHover(name)}
      onMouseLeave={() => onHover(null)}
    >
      <div className="slot-top">
        <span className="slot-name">{name || '?'}</span>
        {group && <span className="slot-group">Grp {group}</span>}
      </div>
      {name && pct != null && pctBar(pct, color)}
    </div>
  )
}

function MatchupCard({ matchup, groups, knockout, stage, highlight, onHover }) {
  // Try to resolve the most likely team for each slot from simulation data
  const teamsByGroup = useMemo(() => {
    const map = {}
    if (!groups) return map
    Object.entries(groups).forEach(([grp, data]) => {
      if (data.teams.length >= 2) {
        map[`1${grp}`] = data.teams[0]  // sim winner
        map[`2${grp}`] = data.teams[1]  // sim runner-up
      }
    })
    return map
  }, [groups])

  const getTeam = (slot) => {
    if (slot === '3rd') return { name: 'Best 3rd', group: '?' }
    return teamsByGroup[slot] || { name: slot, group: '?' }
  }

  const getKOPct = (teamName, stageKey) => {
    if (!knockout || !teamName) return null
    const entry = knockout.find(t => t.name === teamName)
    if (!entry) return null
    const colMap = { r32: 'r32_pct', r16: 'r16_pct', qf: 'qf_pct', sf: 'sf_pct', final: 'final_pct', champion: 'champion_pct' }
    return entry[colMap[stageKey]] ?? null
  }

  const home = getTeam(matchup.home.slot)
  const away = getTeam(matchup.away.slot)
  const homePct = getKOPct(home.name, stage)
  const awayPct = getKOPct(away.name, stage)

  return (
    <div className="matchup-card">
      <div className="match-id">{matchup.id}</div>
      <TeamSlot name={home.name} group={home.group} pct={homePct} stage={stage} highlight={highlight} onHover={onHover} />
      <div className="vs-divider">vs</div>
      <TeamSlot name={away.name} group={away.group} pct={awayPct} stage={stage} highlight={highlight} onHover={onHover} />
    </div>
  )
}

function KOColumn({ title, teams, stage, knockout, highlight, onHover }) {
  if (!teams || teams.length === 0) return null
  const colMap = { r32: 'r32_pct', r16: 'r16_pct', qf: 'qf_pct', sf: 'sf_pct', final: 'final_pct', champion: 'champion_pct' }

  return (
    <div className="ko-column">
      <div className="ko-col-title">{title}</div>
      {teams.map(t => {
        const entry = knockout?.find(k => k.name === t)
        const pct = entry?.[colMap[stage]] ?? 0
        const color = pct > 50 ? '#3fb950' : pct > 25 ? '#58a6ff' : pct > 10 ? '#d29922' : '#8b949e'
        const isHL = highlight === t
        return (
          <div
            key={t}
            className={`ko-team-chip ${isHL ? 'highlighted' : ''}`}
            onMouseEnter={() => onHover(t)}
            onMouseLeave={() => onHover(null)}
          >
            <span className="chip-name">{t}</span>
            {pctBar(pct, color)}
          </div>
        )
      })}
    </div>
  )
}

export default function BracketView({ groups, knockout }) {
  const [highlight, setHighlight] = useState(null)
  const [view, setView] = useState('r32') // 'r32' | 'odds'

  // For odds view — full probability table sorted by champion_pct
  const sortedKO = useMemo(() =>
    knockout ? [...knockout].sort((a, b) => b.champion_pct - a.champion_pct) : []
  , [knockout])

  // Top teams by each stage for the bracket odds view
  const topByStage = (stageKey, n = 8) =>
    [...(knockout || [])].sort((a, b) => b[stageKey] - a[stageKey]).slice(0, n).map(t => t.name)

  return (
    <div className="bracket-view">
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 className="section-title">Predicted Bracket</h2>
        <p className="section-sub">
          R32 matchups follow the official FIFA schedule (Matches 73–88).
          Team shown is the most likely group finisher. Bar = probability of advancing from that round.
        </p>
      </div>

      <div className="bracket-tabs" style={{ marginBottom: '1rem' }}>
        {[['r32', '🗓 R32 Matchups'], ['odds', '📊 Stage Probabilities']].map(([k, label]) => (
          <button key={k} className={`ctrl-btn ${view === k ? 'active' : ''}`} onClick={() => setView(k)}>
            {label}
          </button>
        ))}
      </div>

      {view === 'r32' && (
        <div className="r32-grid">
          <div className="bracket-half">
            <div className="half-label">Left Half (M73–M80)</div>
            {BRACKET_LEFT.map(id => {
              const m = R32_STRUCTURE.find(x => x.id === id)
              return (
                <MatchupCard key={id} matchup={m} groups={groups} knockout={knockout}
                  stage="r16" highlight={highlight} onHover={setHighlight} />
              )
            })}
          </div>
          <div className="bracket-half">
            <div className="half-label">Right Half (M81–M88)</div>
            {BRACKET_RIGHT.map(id => {
              const m = R32_STRUCTURE.find(x => x.id === id)
              return (
                <MatchupCard key={id} matchup={m} groups={groups} knockout={knockout}
                  stage="r16" highlight={highlight} onHover={setHighlight} />
              )
            })}
          </div>
        </div>
      )}

      {view === 'odds' && (
        <div className="odds-view">
          <div className="ko-stages-row">
            {[
              { key: 'r32_pct',      label: 'Qualified (R32)', n: 16 },
              { key: 'r16_pct',      label: 'R16',             n: 12 },
              { key: 'qf_pct',       label: 'Quarterfinals',   n: 8  },
              { key: 'sf_pct',       label: 'Semifinals',      n: 6  },
              { key: 'final_pct',    label: 'Final',           n: 4  },
              { key: 'champion_pct', label: '🏆 Champion',     n: 4  },
            ].map(({ key, label, n }) => (
              <KOColumn key={key} title={label}
                teams={topByStage(key, n)} stage={key.replace('_pct','')}
                knockout={knockout} highlight={highlight} onHover={setHighlight} />
            ))}
          </div>

          <div className="card" style={{ marginTop: '1.5rem', padding: 0, overflow: 'hidden' }}>
            <div style={{ overflowX: 'auto' }}>
              <table className="sim-table">
                <thead>
                  <tr>
                    <th>#</th><th>Team</th><th>Grp</th>
                    <th className="num">R32%</th>
                    <th className="num">R16%</th>
                    <th className="num">QF%</th>
                    <th className="num">SF%</th>
                    <th className="num">Final%</th>
                    <th className="num">🏆%</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedKO.map((t, i) => (
                    <tr key={t.name}
                      onMouseEnter={() => setHighlight(t.name)}
                      onMouseLeave={() => setHighlight(null)}
                      style={{ background: highlight === t.name ? 'var(--surface2)' : undefined }}
                    >
                      <td className="text-muted" style={{ fontSize: '0.78rem' }}>{i+1}</td>
                      <td style={{ fontWeight: 500 }}>{t.name}</td>
                      <td className="text-muted">{t.group}</td>
                      <td className="num">{t.r32_pct.toFixed(1)}%</td>
                      <td className="num">{t.r16_pct.toFixed(1)}%</td>
                      <td className="num">{t.qf_pct.toFixed(1)}%</td>
                      <td className="num">{t.sf_pct.toFixed(1)}%</td>
                      <td className="num">{t.final_pct.toFixed(1)}%</td>
                      <td className="num" style={{ fontWeight: 700, color: t.champion_pct >= 8 ? '#ffd700' : t.champion_pct >= 4 ? '#58a6ff' : undefined }}>
                        {t.champion_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.35rem; }
        .section-sub { color: var(--text-muted); font-size: 0.875rem; }
        .bracket-tabs { display: flex; gap: 0.5rem; }
        .ctrl-btn {
          background: var(--surface2); border: 1px solid var(--border);
          border-radius: 4px; color: var(--text-muted); cursor: pointer;
          font-size: 0.82rem; padding: 4px 12px; transition: all 0.12s;
        }
        .ctrl-btn:hover  { color: var(--text); }
        .ctrl-btn.active { background: var(--blue-dim); border-color: var(--blue); color: var(--blue); }

        .r32-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        @media (max-width: 800px) { .r32-grid { grid-template-columns: 1fr; } }

        .bracket-half { display: flex; flex-direction: column; gap: 0.5rem; }
        .half-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }

        .matchup-card {
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 6px; padding: 0.6rem 0.75rem;
        }
        .match-id { font-size: 0.68rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 600; letter-spacing: 0.05em; }
        .team-slot { padding: 4px 6px; border-radius: 4px; transition: background 0.12s; cursor: default; }
        .team-slot.highlighted { background: var(--blue-dim); }
        .slot-top { display: flex; align-items: baseline; gap: 0.5rem; }
        .slot-name { font-size: 0.88rem; font-weight: 500; }
        .slot-group { font-size: 0.68rem; color: var(--text-muted); }
        .vs-divider { font-size: 0.7rem; color: var(--text-muted); text-align: center; padding: 2px 0; }

        .ko-stages-row { display: flex; gap: 0.75rem; overflow-x: auto; padding-bottom: 0.5rem; }
        .ko-column { min-width: 140px; flex: 1; }
        .ko-col-title { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600; }
        .ko-team-chip {
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 5px; padding: 5px 8px; margin-bottom: 0.35rem;
          cursor: default; transition: background 0.12s;
        }
        .ko-team-chip.highlighted { background: var(--blue-dim); border-color: var(--blue); }
        .chip-name { font-size: 0.82rem; font-weight: 500; display: block; margin-bottom: 2px; }
        .num { text-align: right; }
      `}</style>
    </div>
  )
}
