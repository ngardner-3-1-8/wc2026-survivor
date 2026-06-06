import { useState, useEffect, useMemo } from 'react'

// The Odds API — free tier (500 requests/month)
// Users must supply their own key. We fetch live odds client-side so no key is baked in.
const ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

function americanToDecimal(american) {
  if (!american) return null
  if (american > 0) return (american / 100) + 1
  return (100 / Math.abs(american)) + 1
}

function impliedPct(american) {
  if (!american) return null
  if (american > 0) return 100 / (american + 100) * 100
  return Math.abs(american) / (Math.abs(american) + 100) * 100
}

function edge(modelPct, marketAmerican) {
  if (!marketAmerican || !modelPct) return null
  const mktPct = impliedPct(marketAmerican)
  return (modelPct - mktPct).toFixed(1)
}

function EdgeBadge({ val }) {
  const n = parseFloat(val)
  if (isNaN(n)) return <span className="text-muted">—</span>
  if (n >= 5)  return <span className="edge-badge edge-great">+{n}%</span>
  if (n >= 2)  return <span className="edge-badge edge-good">+{n}%</span>
  if (n >= 0)  return <span className="edge-badge edge-neutral">{n}%</span>
  return <span className="edge-badge edge-bad">{n}%</span>
}

function PctCell({ val, threshold = 55 }) {
  const n = parseFloat(val)
  const color = n >= threshold ? 'var(--green)' : n >= 40 ? 'var(--blue)' : 'var(--text-muted)'
  return <span style={{ color, fontWeight: n >= threshold ? 600 : 400 }}>{val}%</span>
}

// ── Market tabs ──────────────────────────────────────────────────────────────

const MARKETS = [
  { id:'moneyline', label:'💰 Moneyline' },
  { id:'totals',    label:'📊 Totals' },
  { id:'teamtot',   label:'⚽ Team Totals' },
  { id:'btts',      label:'🔁 BTTS' },
  { id:'all',       label:'📋 All Markets' },
]

export default function BettingPage({ betting }) {
  const [market, setMarket] = useState('moneyline')
  const [oddsApiKey, setOddsApiKey] = useState(localStorage.getItem('oddsApiKey') || '')
  const [liveOdds, setLiveOdds] = useState(null)
  const [oddsLoading, setOddsLoading] = useState(false)
  const [oddsError, setOddsError] = useState(null)
  const [sortBy, setSortBy] = useState('edge')
  const [minEdge, setMinEdge] = useState(0)
  const [showKeyInput, setShowKeyInput] = useState(false)

  const matches = betting?.group_matches || []

  // Fetch live odds from The Odds API
  async function fetchLiveOdds() {
    if (!oddsApiKey) { setShowKeyInput(true); return }
    setOddsLoading(true)
    setOddsError(null)
    try {
      const url = `${ODDS_API_BASE}/sports/soccer_fifa_world_cup/odds/?apiKey=${oddsApiKey}&regions=us&markets=h2h,totals&oddsFormat=american`
      const res = await fetch(url)
      if (!res.ok) throw new Error(`Odds API error ${res.status}`)
      const data = await res.json()
      // Build lookup: "HomeTeam|AwayTeam" -> odds object
      const lookup = {}
      data.forEach(game => {
        const key = `${game.home_team}|${game.away_team}`
        lookup[key] = game
        // Also try reverse for flexibility
        lookup[`${game.away_team}|${game.home_team}`] = game
      })
      setLiveOdds(lookup)
    } catch (e) {
      setOddsError(e.message)
    } finally {
      setOddsLoading(false)
    }
  }

  function saveKey(k) {
    setOddsApiKey(k)
    localStorage.setItem('oddsApiKey', k)
    setShowKeyInput(false)
  }

  // Enrich matches with live odds where available
  const enriched = useMemo(() => {
    return matches.map(m => {
      const key = `${m.home}|${m.away}`
      const live = liveOdds?.[key]
      let liveHomeML = null, liveAwayML = null, liveDrawML = null
      let liveOver25 = null, liveUnder25 = null

      if (live) {
        const bestBook = live.bookmakers?.[0]
        if (bestBook) {
          const h2h = bestBook.markets?.find(mk => mk.key === 'h2h')
          if (h2h) {
            const ho = h2h.outcomes?.find(o => o.name === m.home)
            const ao = h2h.outcomes?.find(o => o.name === m.away)
            const dr = h2h.outcomes?.find(o => o.name === 'Draw')
            liveHomeML = ho?.price
            liveAwayML = ao?.price
            liveDrawML = dr?.price
          }
          const tot = bestBook.markets?.find(mk => mk.key === 'totals')
          if (tot) {
            const ov = tot.outcomes?.find(o => o.name === 'Over')
            const un = tot.outcomes?.find(o => o.name === 'Under')
            liveOver25 = ov?.price
            liveUnder25 = un?.price
          }
        }
      }

      return {
        ...m,
        liveHomeML, liveAwayML, liveDrawML,
        liveOver25, liveUnder25,
        edgeHomeML:  edge(m.model_home_wp, liveHomeML),
        edgeDrawML:  edge(m.model_draw_p,  liveDrawML),
        edgeAwayML:  edge(m.model_away_wp, liveAwayML),
        edgeOver25:  edge(m.model_over_25, liveOver25),
        edgeUnder25: edge(m.model_under_25,liveUnder25),
        hasLive: !!live,
      }
    })
  }, [matches, liveOdds])

  // Best bets: sort by edge, filter by min edge
  const bestBets = useMemo(() => {
    const bets = []
    enriched.forEach(m => {
      const add = (label, modelPct, liveML, edgeVal) => {
        const n = parseFloat(edgeVal)
        if (!isNaN(n) && n >= minEdge) {
          bets.push({ match:`${m.home} vs ${m.away}`, group:m.group,
            label, modelPct, liveML, edge:n,
            xgf:m.model_xgf, xga:m.model_xga })
        }
      }
      add(`${m.home} ML`,   m.model_home_wp,  m.liveHomeML,  m.edgeHomeML)
      add(`Draw`,           m.model_draw_p,   m.liveDrawML,  m.edgeDrawML)
      add(`${m.away} ML`,   m.model_away_wp,  m.liveAwayML,  m.edgeAwayML)
      add(`Over 2.5`,       m.model_over_25,  m.liveOver25,  m.edgeOver25)
      add(`Under 2.5`,      m.model_under_25, m.liveUnder25, m.edgeUnder25)
    })
    return bets.sort((a,b) => b.edge - a.edge)
  }, [enriched, minEdge])

  return (
    <div className="betting-view">
      <div style={{ marginBottom:'1.25rem' }}>
        <h2 className="section-title">Betting Intelligence</h2>
        <p className="section-sub">
          Model implied probabilities vs live market odds. Edge = model% − market implied%.
          Positive edge = model sees value. Connect The Odds API for live lines.
        </p>
      </div>

      {/* Odds API connection bar */}
      <div className="card odds-connect-bar">
        <div style={{ display:'flex', alignItems:'center', gap:'0.75rem', flexWrap:'wrap' }}>
          <div>
            <div style={{ fontSize:'0.78rem', fontWeight:600, marginBottom:2 }}>Live Odds</div>
            <div style={{ fontSize:'0.72rem', color:'var(--text-muted)' }}>
              {liveOdds ? `✓ ${Object.keys(liveOdds).length/2} games loaded` : 'Not connected — model odds only'}
            </div>
          </div>
          {showKeyInput ? (
            <div style={{ display:'flex', gap:6, flex:1, minWidth:280 }}>
              <input
                className="key-input"
                placeholder="Paste your Odds API key…"
                defaultValue={oddsApiKey}
                onKeyDown={e => e.key==='Enter' && saveKey(e.target.value)}
              />
              <button className="ctrl-btn active" onClick={e => saveKey(e.target.previousSibling.value)}>Save</button>
            </div>
          ) : (
            <button className="ctrl-btn" onClick={fetchLiveOdds} disabled={oddsLoading}>
              {oddsLoading ? 'Fetching…' : liveOdds ? '🔄 Refresh Odds' : '🔌 Connect Odds API'}
            </button>
          )}
          <button className="ctrl-btn" style={{ fontSize:'0.72rem' }} onClick={()=>setShowKeyInput(s=>!s)}>
            {showKeyInput ? 'Cancel' : '🔑 Set Key'}
          </button>
          {oddsError && <span style={{ color:'var(--red)', fontSize:'0.78rem' }}>{oddsError}</span>}
          <a href="https://the-odds-api.com" target="_blank" rel="noreferrer"
            style={{ fontSize:'0.72rem', color:'var(--blue)', marginLeft:'auto' }}>
            Get free API key →
          </a>
        </div>
      </div>

      {/* Best bets summary */}
      {liveOdds && bestBets.length > 0 && (
        <div className="card best-bets-card">
          <div style={{ display:'flex', alignItems:'center', gap:'0.75rem', marginBottom:'0.75rem', flexWrap:'wrap' }}>
            <div className="card-title" style={{ margin:0 }}>⭐ Best Bets (Edge ≥ {minEdge}%)</div>
            <div style={{ display:'flex', align:'center', gap:6, marginLeft:'auto' }}>
              <label style={{ fontSize:'0.72rem', color:'var(--text-muted)' }}>Min edge:</label>
              {[0,2,5,8].map(v=>(
                <button key={v} className={`ctrl-btn ${minEdge===v?'active':''}`}
                  style={{ padding:'2px 8px', fontSize:'0.72rem' }}
                  onClick={()=>setMinEdge(v)}>{v}%</button>
              ))}
            </div>
          </div>
          <table className="sim-table">
            <thead><tr>
              <th>Match</th><th>Grp</th><th>Bet</th>
              <th className="num">Model%</th><th className="num">Live ML</th>
              <th className="num">Edge</th><th className="num">xGF</th><th className="num">xGA</th>
            </tr></thead>
            <tbody>
              {bestBets.slice(0,20).map((b,i)=>(
                <tr key={i}>
                  <td style={{ fontWeight:500, fontSize:'0.84rem' }}>{b.match}</td>
                  <td className="text-muted">{b.group}</td>
                  <td>{b.label}</td>
                  <td className="num"><PctCell val={b.modelPct} /></td>
                  <td className="num" style={{ color:'var(--yellow)' }}>{b.liveML>0?`+${b.liveML}`:b.liveML}</td>
                  <td className="num"><EdgeBadge val={b.edge} /></td>
                  <td className="num text-muted">{b.xgf}</td>
                  <td className="num text-muted">{b.xga}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Market tabs */}
      <div style={{ display:'flex', gap:'0.4rem', marginBottom:'1rem', flexWrap:'wrap', marginTop:'1.25rem' }}>
        {MARKETS.map(m=>(
          <button key={m.id} className={`ctrl-btn ${market===m.id?'active':''}`} onClick={()=>setMarket(m.id)}>{m.label}</button>
        ))}
      </div>

      {/* Full match table */}
      <div className="card" style={{ padding:0, overflow:'hidden' }}>
        <div style={{ overflowX:'auto' }}>
          <table className="sim-table">
            <thead>
              <tr>
                <th>Home</th><th>Away</th><th>Grp</th>
                {(market==='moneyline'||market==='all') && <>
                  <th className="num">H Win%</th><th className="num">Draw%</th><th className="num">A Win%</th>
                  <th className="num">Mdl H ML</th><th className="num">Mdl D ML</th><th className="num">Mdl A ML</th>
                  {liveOdds && <><th className="num">Live H ML</th><th className="num">Live D ML</th><th className="num">Live A ML</th><th className="num">H Edge</th><th className="num">A Edge</th></>}
                </>}
                {(market==='totals'||market==='all') && <>
                  <th className="num">xG Tot</th>
                  <th className="num">O2.5%</th><th className="num">U2.5%</th><th className="num">O3.5%</th>
                  {liveOdds && <><th className="num">Live O2.5</th><th className="num">O Edge</th></>}
                </>}
                {(market==='teamtot'||market==='all') && <>
                  <th className="num">H O1.5%</th><th className="num">A O1.5%</th>
                </>}
                {(market==='btts'||market==='all') && <>
                  <th className="num">BTTS Yes%</th><th className="num">BTTS No%</th>
                </>}
              </tr>
            </thead>
            <tbody>
              {enriched.map((m,i)=>(
                <tr key={i} style={{ background: m.hasLive?undefined:'inherit' }}>
                  <td style={{ fontWeight:500 }}>{m.home}</td>
                  <td style={{ fontWeight:500 }}>{m.away}</td>
                  <td className="text-muted">{m.group}</td>
                  {(market==='moneyline'||market==='all') && <>
                    <td className="num"><PctCell val={m.model_home_wp} /></td>
                    <td className="num text-muted">{m.model_draw_p}%</td>
                    <td className="num"><PctCell val={m.model_away_wp} /></td>
                    <td className="num text-muted">{m.model_ml_home>0?`+${m.model_ml_home}`:m.model_ml_home}</td>
                    <td className="num text-muted">{m.model_ml_draw>0?`+${m.model_ml_draw}`:m.model_ml_draw}</td>
                    <td className="num text-muted">{m.model_ml_away>0?`+${m.model_ml_away}`:m.model_ml_away}</td>
                    {liveOdds && <>
                      <td className="num" style={{ color:'var(--yellow)' }}>{m.liveHomeML!=null?(m.liveHomeML>0?`+${m.liveHomeML}`:m.liveHomeML):'—'}</td>
                      <td className="num" style={{ color:'var(--yellow)' }}>{m.liveDrawML!=null?(m.liveDrawML>0?`+${m.liveDrawML}`:m.liveDrawML):'—'}</td>
                      <td className="num" style={{ color:'var(--yellow)' }}>{m.liveAwayML!=null?(m.liveAwayML>0?`+${m.liveAwayML}`:m.liveAwayML):'—'}</td>
                      <td className="num"><EdgeBadge val={m.edgeHomeML} /></td>
                      <td className="num"><EdgeBadge val={m.edgeAwayML} /></td>
                    </>}
                  </>}
                  {(market==='totals'||market==='all') && <>
                    <td className="num" style={{ fontWeight:500 }}>{m.model_xg_total}</td>
                    <td className="num"><PctCell val={m.model_over_25} threshold={55}/></td>
                    <td className="num"><PctCell val={m.model_under_25} threshold={55}/></td>
                    <td className="num text-muted">{m.model_over_35}%</td>
                    {liveOdds && <>
                      <td className="num" style={{ color:'var(--yellow)' }}>{m.liveOver25!=null?(m.liveOver25>0?`+${m.liveOver25}`:m.liveOver25):'—'}</td>
                      <td className="num"><EdgeBadge val={m.edgeOver25} /></td>
                    </>}
                  </>}
                  {(market==='teamtot'||market==='all') && <>
                    <td className="num"><PctCell val={m.model_home_over_15} threshold={60}/></td>
                    <td className="num"><PctCell val={m.model_away_over_15} threshold={60}/></td>
                  </>}
                  {(market==='btts'||market==='all') && <>
                    <td className="num"><PctCell val={m.model_btts_yes} threshold={55}/></td>
                    <td className="num text-muted">{m.model_btts_no}%</td>
                  </>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop:'0.75rem', fontSize:'0.72rem', color:'var(--text-muted)' }}>
        Edge = model implied % − market implied %. Green ≥5pp, Yellow 2–5pp. Model probabilities from {betting ? matches.length : 0} simulated group matches.
        Live odds via <a href="https://the-odds-api.com" target="_blank" rel="noreferrer" style={{ color:'var(--blue)' }}>The Odds API</a>.
        Not financial advice. Always gamble responsibly.
      </div>

      <style>{`
        .section-title { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; margin-bottom:0.35rem; }
        .section-sub { color:var(--text-muted); font-size:0.875rem; }
        .ctrl-btn { background:var(--surface2); border:1px solid var(--border); border-radius:4px; color:var(--text-muted); cursor:pointer; font-size:0.82rem; padding:4px 12px; transition:all .12s; }
        .ctrl-btn:hover { color:var(--text); }
        .ctrl-btn.active { background:var(--blue-dim); border-color:var(--blue); color:var(--blue); }
        .ctrl-btn:disabled { opacity:0.5; cursor:not-allowed; }
        .odds-connect-bar { padding:0.75rem 1rem; margin-bottom:1rem; }
        .key-input { flex:1; background:var(--surface2); border:1px solid var(--border); border-radius:4px; color:var(--text); padding:4px 10px; font-size:0.82rem; }
        .best-bets-card { padding:0.85rem 1rem 1rem; margin-bottom:0; }
        .edge-badge { font-size:0.72rem; font-weight:700; padding:2px 6px; border-radius:4px; }
        .edge-great { background:#1a4a24; color:#3fb950; }
        .edge-good  { background:#3d2e00; color:#d29922; }
        .edge-neutral { background:var(--surface2); color:var(--text-muted); }
        .edge-bad   { background:#4a1c1c; color:#f85149; }
        .num { text-align:right; }
        .text-muted { color:var(--text-muted); }
      `}</style>
    </div>
  )
}
