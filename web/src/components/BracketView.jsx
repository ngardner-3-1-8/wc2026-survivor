import { useState } from 'react'

// ── Source match labels for each knockout round card ────────────────────────
const MATCH_SOURCES = {
  // R16 match → which two R32 matches feed it
  M89: 'W(M74) · W(M77)', M90: 'W(M73) · W(M75)',
  M93: 'W(M83) · W(M84)', M94: 'W(M81) · W(M82)',
  M91: 'W(M76) · W(M78)', M92: 'W(M79) · W(M80)',
  M95: 'W(M86) · W(M88)', M96: 'W(M85) · W(M87)',
  // QF
  M97: 'W(M89) · W(M90)',  M98: 'W(M93) · W(M94)',
  M99: 'W(M91) · W(M92)',  M100:'W(M95) · W(M96)',
  // SF
  M101:'W(M97) · W(M98)',  M102:'W(M99) · W(M100)',
  // Final
  M104:'W(M101) · W(M102)',
}

// ── Layout constants ─────────────────────────────────────────────────────────
const CARD_W   = 190   // match card width  (px)
const CARD_H   = 72    // match card height (px)
const CARD_GAP = 8     // vertical gap between sibling cards
const COL_GAP  = 48    // horizontal gap between round columns
const ROUNDS   = ['r32','r16','qf','sf','final']
const ROUND_LABELS = { r32:'Round of 32', r16:'Round of 16', qf:'Quarterfinals', sf:'Semifinals', final:'Final' }

// ── Compute vertical positions ───────────────────────────────────────────────
// In a balanced binary bracket each round's cards are vertically centred
// between the two children that feed them.
function computePositions(nLeaves) {
  // nLeaves = 16 (R32 matches). Each round halves the count.
  // R32 slot height = CARD_H + CARD_GAP
  const slotH = CARD_H + CARD_GAP
  const positions = {}

  // R32: evenly spaced
  positions.r32 = Array.from({ length: nLeaves }, (_, i) => i * slotH)

  // Each subsequent round: centre of pairs
  let prev = positions.r32
  for (const round of ['r16','qf','sf','final']) {
    const cur = []
    for (let i = 0; i < prev.length; i += 2) {
      cur.push((prev[i] + prev[i + 1]) / 2)
    }
    positions[round] = cur
    prev = cur
  }
  return positions
}

// ── SVG connector lines between rounds ──────────────────────────────────────
function Connectors({ fromPositions, toPositions, fromX, toX }) {
  const lines = []
  for (let j = 0; j < toPositions.length; j++) {
    const top = fromPositions[j * 2]
    const bot = fromPositions[j * 2 + 1]
    const mid = toPositions[j]
    const fromCY = CARD_H / 2
    const cx1 = fromX + CARD_W
    const cx2 = toX
    const midX = (cx1 + cx2) / 2

    lines.push(
      // top child → midpoint
      <path key={`t${j}`}
        d={`M${cx1},${top+fromCY} H${midX} V${mid+fromCY} H${cx2}`}
        fill="none" stroke="var(--border)" strokeWidth={1.5} />,
      // bottom child → midpoint
      <path key={`b${j}`}
        d={`M${cx1},${bot+fromCY} H${midX} V${mid+fromCY}`}
        fill="none" stroke="var(--border)" strokeWidth={1.5} />
    )
  }
  return <>{lines}</>
}

// ── Match card ───────────────────────────────────────────────────────────────
function MatchCard({ match, highlight, onHover, showChampPct, x, y }) {
  if (!match) return null
  const { home, away } = match
  const hWin = home.win_pct >= away.win_pct

  const TeamRow = ({ t, isWinner }) => (
    <div
      style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '2px 6px', borderRadius: 3,
        background: highlight === t.name ? 'var(--blue-dim)'
          : isWinner ? 'rgba(63,185,80,0.07)' : 'transparent',
        cursor: 'default', transition: 'background .1s',
      }}
      onMouseEnter={() => onHover(t.name)}
      onMouseLeave={() => onHover(null)}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 0 }}>
        <span style={{
          fontWeight: isWinner ? 700 : 400, fontSize: '0.8rem',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          color: isWinner ? 'var(--text)' : 'var(--text-muted)',
        }}>{t.name}</span>
        {t.is_third && (
          <span style={{ fontSize: '0.6rem', color: 'var(--yellow)', border: '1px solid var(--yellow)', borderRadius: 3, padding: '0 3px', flexShrink: 0 }}>3rd</span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
        {showChampPct && t.champion_pct != null &&
          <span style={{ fontSize: '0.67rem', color: 'var(--yellow)' }}>🏆{t.champion_pct?.toFixed(1)}%</span>}
        <span style={{
          fontSize: '0.75rem', fontWeight: isWinner ? 600 : 400,
          color: isWinner ? 'var(--green)' : 'var(--text-muted)',
        }}>{t.win_pct?.toFixed(0)}%</span>
      </div>
    </div>
  )

  return (
    <div style={{
      position: 'absolute', left: x, top: y,
      width: CARD_W, height: CARD_H,
      background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6,
      padding: '4px 0', boxSizing: 'border-box',
      display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
    }}>
      <div style={{ padding: '0 6px 2px' }}>
        <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em' }}>
          {match.match_id}
        </span>
        {MATCH_SOURCES[match.match_id] && (
          <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', opacity: 0.6, marginLeft: 5 }}>
            {MATCH_SOURCES[match.match_id]}
          </span>
        )}
      </div>
      <div style={{ height: 2, background: 'var(--surface2)', margin: '0 6px' }}>
        <div style={{ height: '100%', background: hWin ? 'var(--green)' : 'var(--blue)', width: `${home.win_pct}%`, transition: 'width .3s' }} />
      </div>
      <TeamRow t={home} isWinner={hWin} />
      <TeamRow t={away} isWinner={!hWin} />
    </div>
  )
}

// ── Third place table ────────────────────────────────────────────────────────
function ThirdPlaceTable({ thirds }) {
  if (!thirds?.length) return null
  return (
    <div className="card third-place-card">
      <div className="card-title">Best 8 Third-Place Teams (Advancing)</div>
      <table className="sim-table">
        <thead><tr>
          <th>#</th><th>Team</th><th>Group</th>
          <th className="num">Avg Pts</th><th className="num">Avg GD</th>
          <th className="num">Avg GF</th><th className="num">Advance%</th>
        </tr></thead>
        <tbody>
          {thirds.map((t, i) => (
            <tr key={t.name}>
              <td className="text-muted">{i + 1}</td>
              <td style={{ fontWeight: 500 }}>{t.name}</td>
              <td className="text-muted">{t.group}</td>
              <td className="num">{t.avg_pts?.toFixed(1)}</td>
              <td className="num" style={{ color: t.avg_gd >= 0 ? 'var(--green)' : 'var(--red)' }}>{t.avg_gd?.toFixed(1)}</td>
              <td className="num">{t.avg_gf?.toFixed(1)}</td>
              <td className="num">
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ flex: 1, height: 5, background: 'var(--surface2)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ width: `${t.advance_pct}%`, height: '100%', background: 'var(--yellow)' }} />
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--yellow)', width: 36, textAlign: 'right' }}>{t.advance_pct}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main bracket view ────────────────────────────────────────────────────────
export default function BracketView({ bracket }) {
  const [highlight, setHighlight] = useState(null)
  const [showChamp, setShowChamp]  = useState(false)
  const [view, setView]            = useState('bracket')

  if (!bracket) return <div className="text-muted" style={{ padding: '2rem' }}>Loading bracket…</div>

  // Build ordered arrays for each round
  const rounds = {
    r32:   bracket.r32   || [],
    r16:   bracket.r16   || [],
    qf:    bracket.qf    || [],
    sf:    bracket.sf    || [],
    final: bracket.final || [],
  }

  const N_R32    = rounds.r32.length   // 16
  const positions = computePositions(N_R32)
  const totalH   = N_R32 * (CARD_H + CARD_GAP) - CARD_GAP + 4
  const colX     = ROUNDS.reduce((acc, r, i) => {
    acc[r] = i * (CARD_W + COL_GAP)
    return acc
  }, {})
  const totalW = colX.final + CARD_W + 4

  return (
    <div className="bracket-view">
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 className="section-title">Predicted Bracket</h2>
        <p className="section-sub">
          Full bracket predicted from simulation · Official FIFA R32 structure (M73–M88) · Green = predicted winner
        </p>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        {[['bracket', '🗓 Full Bracket'], ['thirds', '🥉 3rd Place Rankings']].map(([k, l]) => (
          <button key={k} className={`ctrl-btn ${view === k ? 'active' : ''}`} onClick={() => setView(k)}>{l}</button>
        ))}
        <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--text-muted)', cursor: 'pointer' }}>
          <input type="checkbox" checked={showChamp} onChange={e => setShowChamp(e.target.checked)} />
          Show 🏆 champion%
        </label>
      </div>

      {view === 'thirds' && <ThirdPlaceTable thirds={bracket.third_place_ranking} />}

      {view === 'bracket' && (
        <>
          {/* Champion banner */}
          <div className="champion-banner card" style={{ marginBottom: '1.25rem' }}>
            <span style={{ fontSize: '2.5rem' }}>🏆</span>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Predicted Champion</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: "'Space Grotesk',sans-serif" }}>{bracket.predicted_champion}</div>
            </div>
          </div>

          {/* Round labels row */}
          <div style={{ display: 'flex', gap: 0, marginBottom: 8, paddingLeft: 2 }}>
            {ROUNDS.map(r => (
              <div key={r} style={{ width: CARD_W, marginRight: COL_GAP, fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                {ROUND_LABELS[r]}
              </div>
            ))}
          </div>

          {/* SVG + cards canvas */}
          <div style={{ overflowX: 'auto', overflowY: 'auto', paddingBottom: 8 }}>
            <div style={{ position: 'relative', width: totalW, height: totalH, minWidth: totalW }}>

              {/* Connector SVG underneath cards */}
              <svg style={{ position: 'absolute', inset: 0, overflow: 'visible', pointerEvents: 'none' }}
                width={totalW} height={totalH}>
                {ROUNDS.slice(1).map((round, i) => {
                  const prevRound = ROUNDS[i]
                  return (
                    <Connectors key={round}
                      fromPositions={positions[prevRound]}
                      toPositions={positions[round]}
                      fromX={colX[prevRound]}
                      toX={colX[round]}
                    />
                  )
                })}
              </svg>

              {/* Match cards */}
              {ROUNDS.map(round =>
                rounds[round].map((match, j) => (
                  <MatchCard key={match.match_id}
                    match={match}
                    highlight={highlight}
                    onHover={setHighlight}
                    showChampPct={showChamp}
                    x={colX[round]}
                    y={positions[round][j]}
                  />
                ))
              )}

            </div>
          </div>
        </>
      )}

      <style>{`
        .bracket-view { padding-bottom: 2rem; }
        .section-title { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; margin-bottom:0.35rem; }
        .section-sub { color:var(--text-muted); font-size:0.875rem; }
        .ctrl-btn { background:var(--surface2); border:1px solid var(--border); border-radius:4px; color:var(--text-muted); cursor:pointer; font-size:0.82rem; padding:4px 12px; transition:all .12s; }
        .ctrl-btn:hover { color:var(--text); }
        .ctrl-btn.active { background:var(--blue-dim); border-color:var(--blue); color:var(--blue); }
        .champion-banner { display:flex; align-items:center; gap:1rem; padding:1rem 1.5rem; background:linear-gradient(135deg,#21262d 0%,#1a2a1a 100%); border-color:#3fb95044; }
        .third-place-card { padding:1rem; }
        .num { text-align:right; }
        .text-muted { color:var(--text-muted); }
      `}</style>
    </div>
  )
}
