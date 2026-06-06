import { useState } from 'react'

const ROUNDS = ['r32','r16','qf','sf','final']
const ROUND_LABELS = { r32:'Round of 32', r16:'Round of 16', qf:'Quarterfinals', sf:'Semifinals', final:'Final' }

function WinBar({ p1, p2, name1, name2 }) {
  const c1 = p1 >= p2 ? '#3fb950' : '#58a6ff'
  const c2 = p2 > p1  ? '#3fb950' : '#58a6ff'
  return (
    <div style={{ display:'flex', height:6, borderRadius:3, overflow:'hidden', margin:'4px 0' }}>
      <div style={{ width:`${p1}%`, background:c1 }} title={`${name1} ${p1}%`} />
      <div style={{ width:`${p2}%`, background:c2 }} title={`${name2} ${p2}%`} />
    </div>
  )
}

function MatchCard({ match, highlight, onHover, showChampPct }) {
  const { home, away } = match
  const hWin = home.win_pct >= away.win_pct

  const TeamRow = ({ t, isWinner }) => (
    <div
      className={`bracket-team ${highlight===t.name?'hl':''} ${isWinner?'winner':''}`}
      onMouseEnter={()=>onHover(t.name)} onMouseLeave={()=>onHover(null)}
    >
      <div style={{ display:'flex', alignItems:'baseline', gap:6 }}>
        <span style={{ fontWeight: isWinner?700:400, fontSize:'0.88rem' }}>{t.name}</span>
        {t.is_third && <span style={{ fontSize:'0.65rem', color:'var(--yellow)', border:'1px solid var(--yellow)', borderRadius:3, padding:'0 4px' }}>3rd</span>}
        {t.group && !t.is_third && <span style={{ fontSize:'0.68rem', color:'var(--text-muted)' }}>Grp {t.group}</span>}
      </div>
      <div style={{ display:'flex', alignItems:'center', gap:6, marginTop:2 }}>
        <span style={{ fontSize:'0.78rem', color: isWinner?'var(--green)':'var(--text-muted)', fontWeight:isWinner?600:400 }}>
          {t.win_pct?.toFixed(0)}%
        </span>
        {showChampPct && t.champion_pct != null &&
          <span style={{ fontSize:'0.72rem', color:'var(--yellow)' }}>🏆 {t.champion_pct?.toFixed(1)}%</span>}
      </div>
    </div>
  )

  return (
    <div className="bracket-match-card">
      <div className="match-id-tag">{match.match_id}</div>
      <TeamRow t={home} isWinner={hWin} />
      <WinBar p1={home.win_pct} p2={away.win_pct} name1={home.name} name2={away.name} />
      <TeamRow t={away} isWinner={!hWin} />
    </div>
  )
}

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
          {thirds.map((t,i) => (
            <tr key={t.name}>
              <td className="text-muted">{i+1}</td>
              <td style={{ fontWeight:500 }}>{t.name}</td>
              <td className="text-muted">{t.group}</td>
              <td className="num">{t.avg_pts?.toFixed(1)}</td>
              <td className="num" style={{ color: t.avg_gd>=0?'var(--green)':'var(--red)' }}>{t.avg_gd?.toFixed(1)}</td>
              <td className="num">{t.avg_gf?.toFixed(1)}</td>
              <td className="num">
                <div style={{ display:'flex', alignItems:'center', gap:4 }}>
                  <div style={{ flex:1, height:5, background:'var(--surface2)', borderRadius:2, overflow:'hidden' }}>
                    <div style={{ width:`${t.advance_pct}%`, height:'100%', background:'var(--yellow)' }} />
                  </div>
                  <span style={{ fontSize:'0.75rem', color:'var(--yellow)', width:36, textAlign:'right' }}>{t.advance_pct}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function BracketView({ groups, knockout, bracket }) {
  const [highlight, setHighlight] = useState(null)
  const [showChamp, setShowChamp] = useState(false)
  const [view, setView] = useState('bracket')

  if (!bracket) return <div className="text-muted" style={{ padding:'2rem' }}>Loading bracket…</div>

  const rounds = [
    { key:'r32', data: bracket.r32,   half: 8 },
    { key:'r16', data: bracket.r16,   half: 4 },
    { key:'qf',  data: bracket.qf,    half: 2 },
    { key:'sf',  data: bracket.sf,    half: 1 },
    { key:'final', data: bracket.final, half: 1 },
  ]

  return (
    <div className="bracket-view">
      <div style={{ marginBottom:'1.25rem' }}>
        <h2 className="section-title">Predicted Bracket</h2>
        <p className="section-sub">
          Full tournament bracket predicted from simulation results · R32 follows official FIFA matchup structure (M73–M88) · Green = predicted winner
        </p>
      </div>

      <div style={{ display:'flex', gap:'0.5rem', marginBottom:'1rem', flexWrap:'wrap', alignItems:'center' }}>
        {[['bracket','🗓 Full Bracket'],['thirds','🥉 3rd Place Rankings']].map(([k,l])=>(
          <button key={k} className={`ctrl-btn ${view===k?'active':''}`} onClick={()=>setView(k)}>{l}</button>
        ))}
        <label style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:6, fontSize:'0.82rem', color:'var(--text-muted)', cursor:'pointer' }}>
          <input type="checkbox" checked={showChamp} onChange={e=>setShowChamp(e.target.checked)} />
          Show 🏆 champion%
        </label>
      </div>

      {view === 'thirds' && <ThirdPlaceTable thirds={bracket.third_place_ranking} />}

      {view === 'bracket' && (
        <>
          {/* Champion banner */}
          <div className="champion-banner card">
            <span className="trophy-big">🏆</span>
            <div>
              <div style={{ fontSize:'0.78rem', color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.06em' }}>Predicted Champion</div>
              <div style={{ fontSize:'1.4rem', fontWeight:700, fontFamily:"'Space Grotesk',sans-serif" }}>{bracket.predicted_champion}</div>
            </div>
          </div>

          {/* Bracket rounds */}
          <div className="bracket-scroll">
            {rounds.map(({ key, data }) => (
              <div key={key} className="bracket-round">
                <div className="round-label">{ROUND_LABELS[key]}</div>
                <div className="round-matches">
                  {data?.map(m => (
                    <MatchCard key={m.match_id} match={m}
                      highlight={highlight} onHover={setHighlight}
                      showChampPct={showChamp} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <style>{`
        .section-title { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; margin-bottom:0.35rem; }
        .section-sub { color:var(--text-muted); font-size:0.875rem; }
        .ctrl-btn { background:var(--surface2); border:1px solid var(--border); border-radius:4px; color:var(--text-muted); cursor:pointer; font-size:0.82rem; padding:4px 12px; transition:all .12s; }
        .ctrl-btn:hover { color:var(--text); }
        .ctrl-btn.active { background:var(--blue-dim); border-color:var(--blue); color:var(--blue); }

        .champion-banner { display:flex; align-items:center; gap:1rem; padding:1rem 1.5rem; margin-bottom:1.25rem; background:linear-gradient(135deg, #21262d 0%, #1a2a1a 100%); border-color:#3fb95044; }
        .trophy-big { font-size:2.5rem; }

        .bracket-scroll { display:flex; gap:0.75rem; overflow-x:auto; padding-bottom:1rem; align-items:flex-start; }
        .bracket-round { flex-shrink:0; width:220px; }
        .round-label { font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:0.5rem; padding:0 2px; }
        .round-matches { display:flex; flex-direction:column; gap:0.5rem; }

        .bracket-match-card { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:0.6rem 0.75rem; }
        .match-id-tag { font-size:0.65rem; color:var(--text-muted); font-weight:600; letter-spacing:0.05em; margin-bottom:4px; }
        .bracket-team { padding:3px 5px; border-radius:4px; transition:background .1s; cursor:default; }
        .bracket-team.hl { background:var(--blue-dim); }
        .bracket-team.winner { background:rgba(63,185,80,0.08); }

        .third-place-card { padding:1rem; }
        .num { text-align:right; }
        .text-muted { color:var(--text-muted); }
      `}</style>
    </div>
  )
}
