"""
Daniela Reyes — Assignment 1 Dashboard generator.
Reads data/reviews_predicted.parquet and emits a single, fully self-contained
HTML file (data baked in as JSON) that works offline with no server.

Usage:  python 04_make_dashboard.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "reviews_predicted.parquet")
OUT = os.path.join(ROOT, "Sentiment_Emotion_Dashboard.html")


def emoji_for_emotion(e: str) -> str:
    m = {
        "joy": "😄", "love": "❤️", "surprise": "😮",
        "anger": "😠", "sadness": "😢", "fear": "😨", "neutral": "😐",
    }
    return m.get(e, "😐")


def star_str(rating: float) -> str:
    r = int(round(rating))
    return "★" * r + "☆" * (5 - r)


def main() -> None:
    df = pd.read_parquet(DATA)
    df["matched"] = (df["pred_sentiment"] == df["sentiment_name"]).astype(int)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # timestamp may be ms epoch or s epoch — normalize for display
    ts = df["timestamp"].astype("int64")
    fac = 1.0 if ts.max() < 10**11 else 1e-3
    df["date"] = pd.to_datetime(ts * fac, unit="s").dt.strftime("%Y-%m-%d")

    # ---- aggregate stats (baked so page needs no compute) -----------------
    total = int(len(df))
    n_match = int(df["matched"].sum())
    n_mismatch = total - n_match
    acc = n_match / total if total else 0.0

    # accuracy by true sentiment
    acc_by_sent = {}
    for s in ["negative", "neutral", "positive"]:
        g = df[df["sentiment_name"] == s]
        acc_by_sent[s] = round(float(g["matched"].mean()), 4) if len(g) else None

    # emotion distribution
    emo_counts = df["emotion"].value_counts().to_dict()
    emo_total = sum(emo_counts.values())

    # true-sentiment distribution
    sent_counts = df["sentiment_name"].value_counts().to_dict()
    # model-predicted distribution
    pred_counts = df["pred_sentiment"].value_counts().to_dict()

    # mismatches broken down by (true -> predicted) for the "why" view
    mis = df[df["matched"] == 0].groupby(
        ["sentiment_name", "pred_sentiment"]).size().reset_index(name="n")
    mismatch_pairs = [
        {"true": r["sentiment_name"], "pred": r["pred_sentiment"], "n": int(r["n"])}
        for _, r in mis.iterrows()
    ]

    stats = {
        "total": total, "n_match": n_match, "n_mismatch": n_mismatch,
        "accuracy": round(acc, 4), "acc_by_sent": acc_by_sent,
        "emotion": emo_counts, "emotion_total": emo_total,
        "sent_true": sent_counts, "sent_pred": pred_counts,
        "mismatch_pairs": mismatch_pairs,
        "mean_conf": round(float(df["pred_confidence"].mean()), 4),
        "mean_emo_conf": round(float(df["emotion_score"].mean()), 4),
    }

    # ---- row payload ------------------------------------------------------
    rows = []
    for _, r in df.iterrows():
        e = str(r["emotion"])
        rows.append({
            "date": r["date"],
            "stars": star_str(float(r["rating"])),
            "rating": int(r["rating_int"]),
            "title": str(r["title"]),
            "text": str(r["document"]),
            "true": r["sentiment_name"],
            "pred": r["pred_sentiment"],
            "matched": 1 if r["matched"] else 0,
            "confidence": round(float(r["pred_confidence"]), 3),
            "emotion": e,
            "emotion_icon": emoji_for_emotion(e),
            "emotion_conf": round(float(r["emotion_score"]), 3),
            "verified": bool(r["verified_purchase"]),
            "helpful": int(r["helpful_vote"]),
        })

    payload = json.dumps({"stats": stats, "rows": rows}, ensure_ascii=False)

    html = TEMPLATE.replace("/*__DATA__*/", payload)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {OUT}  ({total} reviews baked in)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment &amp; Emotion Classification of Amazon Reviews</title>
<style>
  :root{
    --ink:#000000;          /* near-black base */
    --surface:#120a0f;      /* black w/ pink cast */
    --card:#1c1118;
    --card2:#241620;
    --line:#3a2230;
    --pink:#ff5fa2;         /* primary hot pink */
    --pink2:#d6336c;        /* deep pink */
    --pink-soft:#7a4057;
    --fg:#f6ecf1;           /* near-white w/ pink tint */
    --muted:#c9a8b8;
    --faint:#8f6f80;
    --pos:#ff5fa2;          /* positive */
    --neu:#d9759f;          /* neutral */
    --neg:#f01a73;          /* negative (brighter magenta) */
    --ok:#e9b7ff;           /* matched accent (soft lilac) */
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{
    background:
      radial-gradient(1200px 600px at 85% -10%, rgba(214,51,108,.18), transparent 60%),
      radial-gradient(900px 500px at -10% 0%, rgba(255,95,162,.10), transparent 55%),
      var(--ink);
    color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    min-height:100vh;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1280px;margin:0 auto;padding:28px 24px 60px}

  header{margin-bottom:26px;border-bottom:1px solid var(--line);padding-bottom:20px}
  header h1{
    font-size:clamp(1.6rem,3.2vw,2.4rem);margin:0;letter-spacing:-.02em;font-weight:800;
    background:linear-gradient(90deg,#fff,var(--pink));
    -webkit-background-clip:text;background-clip:text;color:transparent;
  }
  header .by{color:var(--pink2);font-weight:700;letter-spacing:.14em;text-transform:uppercase;
            font-size:.8rem;margin-top:6px}
  header .meta{color:var(--faint);font-size:.82rem;margin-top:10px}

  /* ---- KPI cards ---- */
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 16px 14px}
  .kpi .lbl{color:var(--faint);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600}
  .kpi .num{font-size:1.9rem;font-weight:800;margin-top:6px;line-height:1}
  .kpi .sub{color:var(--muted);font-size:.78rem;margin-top:6px}
  .kpi.accent{background:linear-gradient(140deg,var(--card2),rgba(214,51,108,.25));border-color:var(--pink2)}
  .num.acc{color:var(--pink)}

  /* ---- layout columns ---- */
  .cols{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}

  /* ---- controls panel ---- */
  .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
  .panel h2{font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;color:var(--pink);
            margin:0 0 14px;font-weight:700}
  .field{margin-bottom:14px}
  .field label{display:block;font-size:.78rem;color:var(--muted);margin-bottom:6px;font-weight:600}
  .seg{display:flex;background:var(--ink);border:1px solid var(--line);border-radius:10px;overflow:hidden}
  .seg button{flex:1;padding:8px 4px;background:transparent;border:none;color:var(--muted);
              font-weight:600;font-size:.8rem;cursor:pointer;transition:.15s}
  .seg button:hover{color:var(--fg)}
  .seg button.active{background:var(--pink2);color:#fff}
  .field select{
    width:100%;padding:8px 10px;background:var(--ink);color:var(--fg);border:1px solid var(--line);
    border-radius:10px;font-size:.85rem
  }
  .search input{
    width:100%;padding:9px 12px;background:var(--ink);color:var(--fg);border:1px solid var(--line);
    border-radius:10px;font-size:.85rem;outline:none}
  .search input:focus{border-color:var(--pink2)}

  /* ---- mini bars in panel ---- */
  .dist .bar-row{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:.78rem}
  .dist .bar-row .k{width:74px;color:var(--muted);text-align:right}
  .dist .track{flex:1;height:9px;background:var(--ink);border-radius:5px;overflow:hidden}
  .dist .fill{height:100%;border-radius:5px}
  .dist .v{width:38px;color:var(--fg);font-variant-numeric:tabular-nums}

  /* ---- mismatch pairs ---- */
  .pairs{display:flex;flex-direction:column;gap:7px}
  .pair{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:.8rem;
        background:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
  .pair .arrow{color:var(--faint)}
  .pair .cnt{background:var(--pink2);color:#fff;border-radius:20px;padding:1px 8px;font-size:.7rem;font-weight:700}

  /* ---- table ---- */
  .table-wrap{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
  .tbar{display:flex;align-items:center;justify-content:space-between;gap:12px;
        padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
  .tbar .count{color:var(--muted);font-size:.85rem}
  .tbar .count b{color:var(--pink)}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  thead th{
    position:sticky;top:0;background:var(--card2);color:var(--pink);text-align:left;
    padding:10px 12px;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;
    border-bottom:1px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap}
  thead th:hover{color:#fff}
  tbody td{padding:11px 12px;border-bottom:1px solid rgba(58,34,48,.6);vertical-align:top}
  tbody tr:hover{background:rgba(255,95,162,.06)}
  tbody tr:last-child td{border-bottom:none}
  .stars{color:var(--pink);white-space:nowrap;letter-spacing:1px}
  .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.72rem;font-weight:700;
        white-space:nowrap}
  .pill.neg{background:rgba(240,26,115,.16);color:#ff7fb0}
  .pill.neu{background:rgba(217,117,159,.16);color:#f0a8c2}
  .pill.pos{background:rgba(255,95,162,.20);color:#ff9ec4}
  .pill.emo{background:var(--ink);color:var(--muted);border:1px solid var(--line)}
  .badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.72rem;font-weight:800}
  .badge.ok{background:#3d2341;color:#e9b7ff}
  .badge.mis{background:#4a1226;color:#ff8fb8;border:1px solid #8f1f47}
  .rev-title{font-weight:600;color:var(--fg);max-width:240px}
  .rev-text{color:var(--muted);max-width:420px;display:-webkit-box;-webkit-line-clamp:2;
            -webkit-box-orient:vertical;overflow:hidden}
  .num{font-variant-numeric:tabular-nums;color:var(--muted)}
  td .vchip{color:var(--faint);font-size:.7rem}
  footer{margin-top:26px;color:var(--faint);font-size:.78rem;text-align:center}

  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:var(--card2);border-radius:6px}
  ::-webkit-scrollbar-track{background:var(--ink)}
  @media(max-width:900px){.cols{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>Sentiment &amp; Emotion Classification of Amazon Reviews</h1>
    <div class="by">Daniela Reyes</div>
    <div class="meta">Amazon 2023 · Gift Cards category · predictions checked against star ratings</div>
  </header>

  <section class="kpis" id="kpis"></section>

  <div class="cols">
    <aside>
      <div class="panel">
        <h2>Filters</h2>
        <div class="field">
          <label>Prediction result</label>
          <div class="seg" id="segMatch">
            <button data-v="all" class="active">All</button>
            <button data-v="match">✓ Matched</button>
            <button data-v="mis">✗ Mismatched</button>
          </div>
        </div>
        <div class="field">
          <label>True sentiment (stars)</label>
          <select id="selTrue">
            <option value="all">All sentiments</option>
          </select>
        </div>
        <div class="field">
          <label>Predicted sentiment</label>
          <select id="selPred">
            <option value="all">All predictions</option>
          </select>
        </div>
        <div class="field">
          <label>Primary emotion</label>
          <select id="selEmo">
            <option value="all">All emotions</option>
          </select>
        </div>
        <div class="field search">
          <label>Search reviews</label>
          <input id="search" type="text" placeholder="Search title or text…">
        </div>
        <button id="reset" class="seg" style="background:var(--ink);color:var(--pink2);width:100%;
          border:1px solid var(--line);border-radius:10px;padding:9px;font-weight:700;cursor:pointer">Reset filters</button>
      </div>

      <div class="panel" style="margin-top:20px">
        <h2>Emotion mix (predicted)</h2>
        <div class="dist" id="emoDist"></div>
      </div>

      <div class="panel" style="margin-top:20px">
        <h2>Where it goes wrong</h2>
        <div class="pairs" id="mismatchPairs"></div>
      </div>
    </aside>

    <section class="table-wrap">
      <div class="tbar">
        <div class="count">Showing <b id="showCount">0</b> of <span id="totalCount">0</span> reviews</div>
      </div>
      <div style="overflow:auto;max-height:72vh">
        <table>
          <thead>
            <tr>
              <th data-k="stars">Rating</th>
              <th data-k="date">Date</th>
              <th data-k="true">True</th>
              <th data-k="pred">Predicted</th>
              <th data-k="matched">Check</th>
              <th data-k="emotion">Emotion</th>
              <th data-k="title">Review</th>
              <th data-k="confidence">Conf</th>
              <th data-k="verified">Verified</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </section>
  </div>

  <footer>Daniela Reyes · Amazon 2023 Gift Cards · predictions checked against star ratings</footer>
</div>

<script>
const D = /*__DATA__*/;

const SENT_FILL = {negative:'var(--neg)', neutral:'var(--neu)', positive:'var(--pos)'};
const EMO_COLOR = {joy:'#ff5fa2', love:'#e268a8', surprise:'#c9a3d6', anger:'#f01a73',
                   sadness:'#7a6db5', fear:'#a06bb5', neutral:'#8f6f80'};

let state = {match:'all', true:'all', pred:'all', emo:'all', q:'', sort:{k:'stars',d:1}};

/* ---------- init ---------- */
function init(){
  buildKpis();
  fillSelect('selTrue', Object.keys(D.stats.sent_true));
  fillSelect('selPred', Object.keys(D.stats.sent_pred));
  fillSelect('selEmo', Object.keys(D.stats.emotion));
  buildEmoDist();
  buildMismatchPairs();
  document.getElementById('totalCount').textContent = D.stats.total;
  bindEvents();
  render();
}

function fillSelect(id, keys){
  const s = document.getElementById(id);
  keys.forEach(k=>{
    const o = document.createElement('option'); o.value=k; o.textContent=k[0].toUpperCase()+k.slice(1);
    s.appendChild(o);
  });
}

function buildKpis(){
  const s = D.stats;
  const sentLabels = {negative:'Negative(1-2★)', neutral:'Neutral(3★)', positive:'Positive(4-5★)'};
  const cards = [
    {lbl:'Reviews', num:s.total, cls:'', sub:'stratified sample'},
    {lbl:'Accuracy', num:(s.accuracy*100).toFixed(1)+'%', cls:'num acc', sub:'model vs star rating'},
    {lbl:'Matched', num:s.n_match, cls:'', sub:'prediction == rating'},
    {lbl:'Mismatched', num:s.n_mismatch, cls:'', sub:'check failed'},
    {lbl:'Neutral acc', num:(s.acc_by_sent.neutral*100??0).toFixed(0)+'%', cls:'', sub:'3★ recall'},
  ];
  document.getElementById('kpis').innerHTML = cards.map(c=>`
    <div class="kpi ${c.cls?'accent':''}"><div class="lbl">${c.lbl}</div><div class="num ${c.cls}">${c.num}</div><div class="sub">${c.sub}</div></div>`).join('');
}

function buildEmoDist(){
  const s = D.stats;
  const max = Math.max(...Object.values(s.emotion));
  document.getElementById('emoDist').innerHTML = Object.entries(s.emotion)
    .sort((a,b)=>b[1]-a[1]).map(([k,v])=>{
      const pct = s.emotion_total? (v/s.emotion_total*100).toFixed(1):'0';
      return `<div class="bar-row"><div class="k">${k}</div>
        <div class="track"><div class="fill" style="width:${v/max*100}%;background:${EMO_COLOR[k]||'var(--pink)'}"></div></div>
        <div class="v">${v} · ${pct}%</div></div>`;
    }).join('');
}

function buildMismatchPairs(){
  const pairs = D.stats.mismatch_pairs;
  if(!pairs.length){document.getElementById('mismatchPairs').innerHTML='<div style="color:var(--muted);font-size:.8rem">No mismatches on this data.</div>';return;}
  document.getElementById('mismatchPairs').innerHTML = pairs.map(p=>`
    <div class="pair"><span class="pill ${p.true}">${p.true}</span>
      <span class="arrow">→</span><span class="pill ${p.pred}">${p.pred}</span>
      <span class="cnt">${p.n}</span></div>`).join('');
}

/* ---------- filtering ---------- */
function matches(r){
  if(state.match==='match' && r.matched!==1) return false;
  if(state.match==='mis'  && r.matched!==0) return false;
  if(state.true!=='all' && r.true!==state.true) return false;
  if(state.pred!=='all' && r.pred!==state.pred) return false;
  if(state.emo!=='all' && r.emotion!==state.emo) return false;
  const q = state.q.trim().toLowerCase();
  if(q && !(r.title.toLowerCase().includes(q) || r.text.toLowerCase().includes(q))) return false;
  return true;
}

function render(){
  let rows = D.rows.filter(matches);
  // sort
  const {k,d} = state.sort;
  rows = rows.sort((a,b)=>{
    let x=a[k], y=b[k];
    if(k==='date'||k==='title'){ x=String(x); y=String(y); }
    if(typeof x==='string' && typeof y==='string'){ const r= x.localeCompare(y); return r*d; }
    return (x-y)*d;
  });
  document.getElementById('showCount').textContent = rows.length;
  const tbody = document.getElementById('tbody');
  if(!rows.length){
    tbody.innerHTML = `<tr><td colspan="9" style="color:var(--faint);text-align:center;padding:30px">No reviews match these filters.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(r=>`
    <tr>
      <td class="stars" title="${r.rating}/5">${r.stars}</td>
      <td class="num">${r.date}</td>
      <td><span class="pill ${r.true}">${r.true}</span></td>
      <td><span class="pill ${r.pred}">${r.pred}</span></td>
      <td><span class="badge ${r.matched?'ok':'mis'}">${r.matched?'✓ matched':'✗ mismatch'}</span></td>
      <td><span class="pill emo" title="conf ${(r.emotion_conf*100).toFixed(0)}%">${r.emotion_icon} ${r.emotion}</span></td>
      <td><div class="rev-title">${esc(r.title)||'—'}</div><div class="rev-text">${esc(r.text)}</div></td>
      <td class="num">${(r.confidence*100).toFixed(0)}%</td>
      <td class="vchip">${r.verified?'✓':'—'}</td>
    </tr>`).join('');
}
function esc(s){return (s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

/* ---------- events ---------- */
function bindEvents(){
  document.querySelectorAll('#segMatch button').forEach(b=>{
    b.onclick=()=>{document.querySelectorAll('#segMatch button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); state.match=b.dataset.v; render();};
  });
  document.getElementById('selTrue').onchange=e=>{state.true=e.target.value;render();};
  document.getElementById('selPred').onchange=e=>{state.pred=e.target.value;render();};
  document.getElementById('selEmo').onchange=e=>{state.emo=e.target.value;render();};
  document.getElementById('search').oninput=e=>{state.q=e.target.value;render();};
  document.getElementById('reset').onclick=reset;
  document.querySelectorAll('th').forEach(th=>{
    th.onclick=()=>{
      const k=th.dataset.k;
      if(state.sort.k===k) state.sort.d*=-1; else state.sort={k,d:1};
      document.querySelectorAll('th').forEach(x=>x.textContent=x.textContent.replace(/ ?[▲▼]$/,''));
      th.textContent += (state.sort.d===1?' ▲':' ▼');
      render();
    };
  });
}
function reset(){
  state={match:'all',true:'all',pred:'all',emo:'all',q:'',sort:{k:'stars',d:1}};
  document.querySelectorAll('#segMatch button').forEach(x=>x.classList.toggle('active',x.dataset.v==='all'));
  ['selTrue','selPred','selEmo'].forEach(id=>document.getElementById(id).value='all');
  document.getElementById('search').value='';
  document.querySelectorAll('th').forEach(x=>x.textContent=x.textContent.replace(/ ?[▲▼]$/,''));
  render();
}

init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
