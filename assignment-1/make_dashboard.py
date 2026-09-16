"""
MBAX 6418 Assignment 1 — Dashboard generator (Steps 3, 4, 7).

Reads the LLM classification results (three-class balanced run + binary batch)
and emits one fully self-contained HTML file with the data baked in as JSON —
works offline, no server, no network.

Includes:
  - KPI cards (headline agreement + per-class accuracy)
  - Descriptive viz: star-rating distribution, true-vs-predicted per class
  - Emotion comparison: LLM primary emotion vs NRC word-list emotion
  - Interactive, filterable review table (correct vs mismatched, by class,
    by emotion, free-text search) with a live count
  - Coherent pink/black theme (Design bar)

Usage:
  python make_dashboard.py
"""
import base64
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "outputs")
OUT_HTML = os.path.join(ROOT, "Sentiment_Emotion_Dashboard.html")

CLASS_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def load(parquet):
    p = os.path.join(OUTDIR, parquet)
    return pd.read_parquet(p) if os.path.exists(p) else None


def row_payload(df, mode):
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "stars": "★" * int(r["rating"]) + "☆" * (5 - int(r["rating"])),
            "rating": int(r["rating"]),
            "title": str(r["title"] or ""),
            "text": str(r["text"] or ""),
            "true": r["true"],
            "pred": r["pred"] if pd.notna(r["pred"]) else None,
            "correct": bool(r["correct"]),
            "emotion_llm": str(r["emotion"]) if pd.notna(r["emotion"]) else None,
            "emotion_nrc": str(r["nrc_emotion"]) if pd.notna(r.get("nrc_emotion")) else None,
            "mode": mode,
        })
    return rows


def per_class(df):
    out = {}
    for c in CLASS_ORDER:
        g = df[df["true"] == c]
        out[c] = {
            "n": int(len(g)),
            "correct": int(g["correct"].sum()) if len(g) else 0,
            "acc": round(float(g["correct"].mean()), 4) if len(g) else 0,
            "pred": {p: int(n) for p, n in g["pred"].value_counts().items()},
        }
    return out


def star_distribution(df):
    return {int(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()}


def main():
    df3 = load("results_three.parquet")
    dfb = load("results_binary.parquet")
    if df3 is None:
        raise SystemExit("results_three.parquet missing — run llm_classify.py first")
    if "nrc_emotion" not in df3.columns:
        raise SystemExit("results_three.parquet has no nrc_emotion — run nrc_emotion.py")

    # ---- three-class (Step 6) aggregates ----
    t3 = {"total": int(len(df3)),
          "agree": int(df3["correct"].sum()),
          "agree_pct": round(float(df3["correct"].mean()), 4) * 100,
          "per_class": per_class(df3),
          "stars": star_distribution(df3),
          }
    # emotion: LLM vs NRC distributions + agreement
    llm = df3["emotion"].dropna()
    nrc = df3["nrc_emotion"].dropna()
    both = df3.dropna(subset=["emotion", "nrc_emotion"])
    t3["emotion_llm"] = llm.value_counts().to_dict()
    t3["emotion_nrc"] = nrc.value_counts().to_dict()
    t3["emotion_agree"] = int((both["emotion"] == both["nrc_emotion"]).sum())
    t3["emotion_both"] = int(len(both))
    t3["emotion_agree_pct"] = round(t3["emotion_agree"] / t3["emotion_both"], 4) * 100 \
        if t3["emotion_both"] else 0
    # NRC emotion by rating class (for the "emotion follows stars" story)
    nrc_by_class = {}
    for c in CLASS_ORDER:
        g = df3[df3["true"] == c]["nrc_emotion"].dropna()
        nrc_by_class[c] = g.value_counts().to_dict()
    t3["nrc_by_class"] = nrc_by_class

    # ---- binary (Step 2) aggregates ----
    b = {}
    if dfb is not None and "nrc_emotion" in dfb.columns:
        b = {"total": int(len(dfb)),
             "agree": int(dfb["correct"].sum()),
             "agree_pct": round(float(dfb["correct"].mean()), 4) * 100,
             "per_class": per_class(dfb),
             }

    payload = {
        "three": t3,
        "binary": b,
        "rows": row_payload(df3, "three") + (row_payload(dfb, "binary") if dfb is not None else []),
    }

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False))
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {OUT_HTML}  ({len(payload['rows'])} rows baked in)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment &amp; Emotion Classification of Amazon Reviews</title>
<style>
  :root{
    --ink:#08060a; --surface:#100b12; --card:#17101c; --card2:#201625;
    --line:#35203f; --pink:#ff5fa2; --pink2:#d6336c; --pink3:#f01a73;
    --fg:#f6ecf3; --muted:#c9a8ba; --faint:#8f6f80;
    --pos:#ff8fc2; --neu:#c99ad6; --neg:#ff5c9e;
    --match:#e9b7ff; --mismatch:#ff8fb8;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{background:radial-gradient(1100px 550px at 88% -8%, rgba(214,51,108,.20), transparent 60%),
       radial-gradient(900px 500px at -8% 0, rgba(255,95,162,.10), transparent 55%), var(--ink);
       color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
       -webkit-font-smoothing:antialiased; min-height:100vh}
  .wrap{max-width:1280px;margin:0 auto;padding:30px 26px 70px}
  header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:26px}
  header h1{font-size:clamp(1.55rem,3vw,2.3rem);margin:0;letter-spacing:-.02em;font-weight:800;
    background:linear-gradient(90deg,#fff,var(--pink) 70%);-webkit-background-clip:text;background-clip:text;color:transparent}
  header .by{color:var(--pink2);font-weight:700;letter-spacing:.16em;text-transform:uppercase;font-size:.78rem;margin-top:8px}
  header .meta{color:var(--faint);font-size:.82rem;margin-top:10px}
  .sechead{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--pink);font-weight:700;margin:30px 0 14px}

  /* KPI */
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
  .kpi .lbl{color:var(--faint);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600}
  .kpi .num{font-size:1.9rem;font-weight:800;margin-top:6px;line-height:1}
  .kpi .sub{color:var(--muted);font-size:.76rem;margin-top:6px}
  .kpi.accent{background:linear-gradient(140deg,var(--card2),rgba(214,51,108,.22));border-color:var(--pink2)}
  .num.pink{color:var(--pink)}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
  .card h2{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--pink);margin:0 0 14px;font-weight:700}

  /* bars */
  .bar-row{display:flex;align-items:center;gap:9px;margin-bottom:8px;font-size:.78rem}
  .bar-row .k{width:86px;color:var(--muted);text-align:right}
  .bar-row .track{flex:1;height:10px;background:var(--ink);border-radius:6px;overflow:hidden}
  .bar-row .fill{height:100%;border-radius:6px}
  .bar-row .v{width:auto;min-width:44px;color:var(--fg);font-variant-numeric:tabular-nums;white-space:nowrap}
  .lg{font-weight:700}

  /* confusion-style table */
  .cm{width:100%;border-collapse:collapse;font-size:.8rem}
  .cm th,.cm td{padding:7px 9px;border:1px solid var(--line);text-align:center}
  .cm th{color:var(--pink);font-weight:700;background:var(--card2)}
  .cm .true{text-align:right;color:var(--muted);font-weight:600}
  .cm .diag{background:rgba(255,95,162,.18);font-weight:700}
  .cm .hot{background:rgba(240,26,115,.28);font-weight:700}
  .cm .sub{font-size:.68rem;color:var(--faint)}

  /* controls */
  .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:18px}
  .panel h2{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--pink);margin:0 0 14px;font-weight:700}
  .field{margin-bottom:13px}
  .field label{display:block;font-size:.76rem;color:var(--muted);margin-bottom:6px;font-weight:600}
  .seg{display:flex;background:var(--ink);border:1px solid var(--line);border-radius:10px;overflow:hidden}
  .seg button{flex:1;padding:8px 4px;background:transparent;border:none;color:var(--muted);font-weight:600;font-size:.8rem;cursor:pointer}
  .seg button:hover{color:var(--fg)}
  .seg button.active{background:var(--pink2);color:#fff}
  .field select{width:100%;padding:8px 10px;background:var(--ink);color:var(--fg);border:1px solid var(--line);border-radius:10px;font-size:.84rem}
  .search input{width:100%;padding:9px 12px;background:var(--ink);color:var(--fg);border:1px solid var(--line);border-radius:10px;font-size:.84rem;outline:none}
  .search input:focus{border-color:var(--pink2)}

  /* table */
  .table-wrap{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
  .tbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
  .tbar .count{color:var(--muted);font-size:.85rem}
  .tbar .count b{color:var(--pink)}
  table.rv{width:100%;border-collapse:collapse;font-size:.84rem}
  thead th{position:sticky;top:0;background:var(--card2);color:var(--pink);text-align:left;padding:10px 12px;
    font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap}
  tbody td{padding:11px 12px;border-bottom:1px solid rgba(53,32,63,.55);vertical-align:top}
  tbody tr:hover{background:rgba(255,95,162,.06)}
  .stars{color:var(--pink);white-space:nowrap;letter-spacing:1px}
  .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.7rem;font-weight:700;white-space:nowrap}
  .pill.neg{background:rgba(255,92,158,.14);color:#ff9ec4}
  .pill.neu{background:rgba(201,154,214,.16);color:#e2c3ee}
  .pill.pos{background:rgba(255,143,194,.18);color:#ffc2dd}
  .badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.72rem;font-weight:800}
  .badge.ok{background:#3d2341;color:var(--match)}
  .badge.mis{background:#4a1226;color:var(--mismatch);border:1px solid #8f1f47}
  .emo{color:var(--muted)}
  .rev-title{font-weight:600;color:var(--fg);max-width:250px}
  .rev-text{color:var(--muted);max-width:340px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .modetag{color:var(--faint);font-size:.7rem;font-style:italic}
  .none{color:var(--faint)}
  footer{margin-top:30px;color:var(--faint);font-size:.78rem;text-align:center}
  ::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--card2);border-radius:6px}::-webkit-scrollbar-track{background:var(--ink)}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>Sentiment &amp; Emotion Classification of Amazon Reviews</h1>
    <div class="by">Daniela Reyes</div>
    <div class="meta">MBAX 6418 · Assignment 1 · Amazon 2023 Gift-Cards reviews · LLM-classified, checked against the star rating</div>
  </header>

  <!-- ===== THREE-CLASS KPIs ===== -->
  <section class="kpis" id="kpix"></section>
  <div class="sechead">Balanced three-class run (50 per class, fixed seed) — how often the LLM matches the rating</div>

  <div class="grid">
    <div class="card"><h2>Star-rating distribution (sample)</h2><div id="stars"></div></div>
    <div class="card"><h2>Classification matrix — true (rating) × predicted (LLM)</h2><div id="cmpan"></div></div>
  </div>

  <div class="grid" style="margin-top:18px">
    <div class="card"><h2>Per-class accuracy</h2><div id="classacc"></div></div>
    <div class="card"><h2>True vs predicted — where mistakes go</h2><div id="truepred"></div></div>
  </div>

  <div class="sechead">Emotion — two independent takes, compared</div>
  <div class="grid">
    <div class="card"><h2>LLM primary emotion</h2><div id="emollm"></div></div>
    <div class="card"><h2>NRC word-list emotion</h2><div id="emonrc"></div></div>
  </div>
  <div class="card" style="margin-top:18px"><h2>NRC word-list emotion by rating class (does word emotion follow the stars?)</h2><div id="nrcclass"></div></div>

  <div class="sechead">Reviews</div>
  <div style="display:grid;grid-template-columns:320px 1fr;gap:18px;align-items:start" class="fcols">
    <aside>
      <div class="panel">
        <h2>Filters</h2>
        <div class="field"><label>Prediction result</label>
          <div class="seg" id="segResult">
            <button data-v="all" class="active">All</button>
            <button data-v="match">✓ Matched</button>
            <button data-v="mis">✗ Mismatched</button>
          </div></div>
        <div class="field"><label>True class (stars)</label><select id="fTrue"><option value="all">All</option></select></div>
        <div class="field"><label>Predicted class</label><select id="fPred"><option value="all">All</option></select></div>
        <div class="field"><label>LLM emotion</label><select id="fEmo"><option value="all">All</option></select></div>
        <div class="field"><label>NRC emotion</label><select id="fNrc"><option value="all">All</option></select></div>
        <div class="field search"><label>Search reviews</label><input id="q" placeholder="Search title or text…"></div>
        <button id="reset" style="width:100%;padding:9px;background:var(--ink);color:var(--pink2);border:1px solid var(--line);border-radius:10px;font-weight:700;cursor:pointer">Reset filters</button>
      </div>
    </aside>
    <section class="table-wrap">
      <div class="tbar"><div class="count">Showing <b id="showCount">0</b> of <span id="totCount">0</span></div>
        <div class="count"><label style="color:var(--faint)">Run:</label>
          <select id="fMode" style="background:var(--ink);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px 8px">
            <option value="all">both runs</option><option value="three">three-class (balanced)</option><option value="binary">binary (first 100)</option>
          </select></div></div>
      <div style="overflow:auto;max-height:70vh">
        <table class="rv"><thead><tr>
          <th data-k="rating">Rating</th><th data-k="true">True</th><th data-k="pred">Predicted</th><th data-k="correct">Check</th>
          <th data-k="emotion_llm">LLM emotion</th><th data-k="emotion_nrc">NRC emotion</th><th data-k="title">Review</th>
        </tr></thead><tbody id="tbody"></tbody></table>
      </div>
    </section>
  </div>

  <footer>Daniela Reyes · MBAX 6418 Assignment 1 · Amazon Reviews '23 (Gift Cards) · single self-contained file · data, not server, drives it</footer>
</div>

<script>
const D = /*__DATA__*/;

/* ---------- theme colors ---------- */
const CLS = {NEGATIVE:'var(--neg)', NEUTRAL:'var(--neu)', POSITIVE:'var(--pos)'};
const CLS_NAME = {NEGATIVE:'negative(1-2★)', NEUTRAL:'neutral(3★)', POSITIVE:'positive(4-5★)'};
const EMO_COLOR = {joy:'#ff5fa2', love:'#e268a8', trust:'#c9a3d6', anticipation:'#b78fd0',
  surprise:'#e0c3f0', sadness:'#7a6db5', fear:'#a06bb5', anger:'#f01a73', disgust:'#8f6f80'};
const ORDER = ['NEGATIVE','NEUTRAL','POSITIVE'];

let state = {result:'all', true:'all', pred:'all', emo:'all', nrc:'all', mode:'all', q:'', sort:{k:'rating',d:-1}};

function bars(container, counts, colorFn, maxv){
  const entries = Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  const max = Math.max(1, maxv || Math.max(...entries.map(e=>e[1]),1));
  const total = entries.reduce((s,e)=>s+e[1],0)||1;
  document.getElementById(container).innerHTML = entries.map(([k,v])=>
    `<div class="bar-row"><div class="k">${k}</div>
      <div class="track"><div class="fill" style="width:${v/max*100}%;background:${colorFn(k)}"></div></div>
      <div class="v">${v} <span class="sub">(${(v/total*100).toFixed(1)}%)</span></div></div>`).join("") ||
    `<div class="none">no data</div>`;
}

/* ---------- KPI cards ---------- */
function buildKpis(){
  const t = D.three;
  const acc = t.agree_pct.toFixed(1);
  const pc = t.per_class;
  const cards =
    `<div class="kpi accent"><div class="lbl">Reviews (balanced)</div><div class="num">${t.total}</div><div class="sub">50 per class · fixed seed</div></div>`+
    `<div class="kpi"><div class="lbl">Agreement vs rating</div><div class="num pink">${acc}%</div><div class="sub">${t.agree}/${t.total} match</div></div>`+
    `<div class="kpi"><div class="lbl">Negative recall</div><div class="num">${(pc.NEGATIVE.acc*100).toFixed(0)}%</div><div class="sub">${pc.NEGATIVE.correct}/${pc.NEGATIVE.n}</div></div>`+
    `<div class="kpi"><div class="lbl">Neutral recall</div><div class="num">${(pc.NEUTRAL.acc*100).toFixed(0)}%</div><div class="sub">${pc.NEUTRAL.correct}/${pc.NEUTRAL.n} ← collapses</div></div>`+
    `<div class="kpi"><div class="lbl">Positive recall</div><div class="num">${(pc.POSITIVE.acc*100).toFixed(0)}%</div><div class="sub">${pc.POSITIVE.correct}/${pc.POSITIVE.n}</div></div>`;
  document.getElementById('kpix').innerHTML = cards;

  // star distribution
  const st = t.stars;
  const smax = Math.max(1,...Object.values(st));
  const stot = Object.values(st).reduce((a,b)=>a+b,0)||1;
  document.getElementById('stars').innerHTML = Object.entries(st).map(([k,v])=>
    `<div class="bar-row"><div class="k">${'★'.repeat(+k)}${'☆'.repeat(5-(+k))}</div>
      <div class="track"><div class="fill" style="width:${v/smax*100}%;background:var(--pink)"></div></div>
      <div class="v">${v} <span class="sub">(${(v/stot*100).toFixed(1)}%)</span></div></div>`).join("");
}

/* ---------- confusion matrix (true × pred) ---------- */
function buildMatrix(){
  const pc = D.three.per_class;
  let html = `<table class="cm"><tr><th class="true">true \\ pred</th>`+ORDER.map(c=>`<th>${c}</th>`).join('')+`<th>recall</th></tr>`;
  ORDER.forEach(t=>{
    const row = pc[t].pred;
    html += `<tr><td class="true">${CLS_NAME[t]}</td>`;
    ORDER.forEach(p=>{
      const v = row[p]||0;
      const cls = p===t ? 'diag' : (v>0 ? 'hot' : '');
      html += `<td class="${cls}">${v}</td>`;
    });
    html += `<td class="sub">${(pc[t].acc*100).toFixed(0)}%</td></tr>`;
  });
  html += `</table><p class="sub" style="margin:8px 0 0">rows = true class from the star rating · diag = correct · pink = mislabelled<br>Both NEGATIVE (96%) and POSITIVE (96%) are caught; NEUTRAL (14%) mostly folds into NEGATIVE.</p>`;
  document.getElementById('cmpan').innerHTML = html;
}

/* ---------- per-class accuracy + true-vs-pred ---------- */
function buildClassAcc(){
  const pc = D.three.per_class;
  const maxn = Math.max(...ORDER.map(c=>pc[c].n),1);
  let html = '';
  ORDER.forEach(c=>{
    const g = pc[c];
    html += `<div class="bar-row"><div class="k">${c}</div>
      <div class="track"><div class="fill" style="width:${(g.acc*100)}%;background:${CLS[c]}"></div></div>
      <div class="v">${(g.acc*100).toFixed(0)}% <span class="sub">${g.correct}/${g.n}</span></div></div>`;
  });
  document.getElementById('classacc').innerHTML = html;
}
function buildTruePred(){
  const pc = D.three.per_class;
  let html='';
  ORDER.forEach(t=>{
    const row = pc[t].pred;
    // label bars by what the true class was predicted as
    ORDER.forEach(p=>{
      const v = row[p]||0;
      const n = pc[t].n;
      html += `<div class="bar-row"><div class="k" style="width:120px">${t} → ${p}</div>
        <div class="track"><div class="fill" style="width:${v/n*100}%;background:${p===t?CLS[t]:'var(--pink3)'}"></div></div>
        <div class="v">${v} <span class="sub">(${(v/n*100).toFixed(0)}%)</span></div></div>`;
    });
  });
  document.getElementById('truepred').innerHTML = html;
}

/* ---------- emotions ---------- */
function buildEmo(){
  bars('emollm', D.three.emotion_llm, k=>EMO_COLOR[k]||'var(--pink)');
  bars('emonrc', D.three.emotion_nrc, k=>EMO_COLOR[k]||'var(--pink)');
  // NRC by class
  const byc = D.three.nrc_by_class;
  let html='';
  ORDER.forEach(c=>{
    const rows = Object.entries(byc[c]||{}).sort((a,b)=>b[1]-a[1]).slice(0,4);
    html += `<div class="bar-row"><div class="k" style="width:120px">${c}</div>
      <div class="track" style="display:flex;gap:2px;height:12px">${
        rows.map(([k,v])=>`<div style="width:${v/(byc[c][rows[0][0]])*100}%;background:${EMO_COLOR[k]||'#888'}"></div>`).join('')}</div>
      <div class="v">${rows.map(([k,v])=>`${k} ${v}`).join(', ')||'—'}</div></div>`;
  });
  document.getElementById('nrcclass').innerHTML = html;

  // emotion agreement banner
  const t = D.three;
  document.querySelector('.grid').insertAdjacentHTML('beforebegin',
    `<div class="card" style="margin-bottom:18px"><h2>Emotion agreement: LLM vs NRC word list</h2>
      <div class="bar-row"><div class="k">agree</div>
        <div class="track"><div class="fill" style="width:${t.emotion_agree_pct}%;background:var(--pink)"></div></div>
        <div class="v">${t.emotion_agree}/${t.emotion_both} = ${t.emotion_agree_pct.toFixed(1)}%</div></div>
      <p class="sub" style="margin:6px 0 0">The two methods rarely pick the same emotion: the word list floods positive reviews with
      anticipation/joy, while the LLM reads meaning (dominant LLM emotion: joy, then anger);
      negatives lean anger/sadness under both.</p></div>`);
}

/* ---------- filtering + table ---------- */
function matchRow(r){
  if(state.result==='match' && !r.correct) return false;
  if(state.result==='mis' && r.correct) return false;
  if(state.true!=='all' && r.true!==state.true) return false;
  if(state.pred!=='all' && r.pred!==state.pred) return false;
  if(state.emo!=='all' && r.emotion_llm!==state.emo) return false;
  if(state.nrc!=='all' && r.emotion_nrc!==state.nrc) return false;
  if(state.mode!=='all' && r.mode!==state.mode) return false;
  const q=state.q.toLowerCase().trim();
  if(q && !(r.title.toLowerCase().includes(q)||r.text.toLowerCase().includes(q))) return false;
  return true;
}
function esc(s){return (s==null?'':String(s)).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function pill(c){return c?`<span class="pill ${c.toLowerCase()}">${c}</span>`:'<span class="none">—</span>';}

function render(){
  let rows = D.rows.filter(matchRow);
  const {k,d}=state.sort;
  const cmp = (a,b)=>{
    if(k==='title'||k==='true'||k==='pred'||k==='emotion_llm'||k==='emotion_nrc'){
      const x=String(a[k]||'').toLowerCase(), y=String(b[k]||'').toLowerCase();
      return (x<y?-1:(x>y?1:0))*d;
    }
    return ((a[k]??0)-(b[k]??0))*d;
  };
  rows.sort(cmp);
  document.getElementById('showCount').textContent = rows.length;
  document.getElementById('totCount').textContent = D.rows.length;
  const tb=document.getElementById('tbody');
  if(!rows.length){tb.innerHTML=`<tr><td colspan="7" style="color:var(--faint);text-align:center;padding:28px">No reviews match these filters.</td></tr>`;return;}
  tb.innerHTML = rows.map(r=>`<tr>
    <td class="stars" title="${r.rating}/5">${r.stars}</td>
    <td>${pill(r.true)}</td>
    <td>${pill(r.pred)}</td>
    <td><span class="badge ${r.correct?'ok':'mis'}">${r.correct?'✓ matched':'✗ mismatch'}</span></td>
    <td><span class="emo">${esc(r.emotion_llm)||'—'}</span></td>
    <td><span class="emo">${esc(r.emotion_nrc)||'—'}</span></td>
    <td><div class="rev-title">${esc(r.title)||''}</div>
        <div class="rev-text">${esc(r.text)}</div><span class="modetag">${r.mode==='three'?'balanced':'first-100'}</span></td>
  </tr>`).join('');
}

function fillSel(id, keys){
  const s=document.getElementById(id);
  keys.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k==='NEGATIVE'?'negative(1-2★)':k==='NEUTRAL'?'neutral(3★)':k==='POSITIVE'?'positive(4-5★)':k;s.appendChild(o);});
}

function bind(){
  document.querySelectorAll('#segResult button').forEach(b=>b.onclick=()=>{
    document.querySelectorAll('#segResult button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); state.result=b.dataset.v; render();});
  document.getElementById('fTrue').onchange=e=>{state.true=e.target.value;render();};
  document.getElementById('fPred').onchange=e=>{state.pred=e.target.value;render();};
  document.getElementById('fEmo').onchange=e=>{state.emo=e.target.value;render();};
  document.getElementById('fNrc').onchange=e=>{state.nrc=e.target.value;render();};
  document.getElementById('fMode').onchange=e=>{state.mode=e.target.value;render();};
  document.getElementById('q').oninput=e=>{state.q=e.target.value;render();};
  document.getElementById('reset').onclick=()=>{
    state={result:'all',true:'all',pred:'all',emo:'all',nrc:'all',mode:'all',q:'',sort:{k:'rating',d:-1}};
    ['fTrue','fPred','fEmo','fNrc','fMode'].forEach(id=>document.getElementById(id).value='all');
    document.getElementById('q').value='';
    document.querySelectorAll('#segResult button').forEach(x=>x.classList.toggle('active',x.dataset.v==='all'));
    render();};
  document.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;
    if(state.sort.k===k) state.sort.d*=-1; else state.sort={k,d:1};
    document.querySelectorAll('th').forEach(x=>x.textContent=x.textContent.replace(/ [▲▼]$/,''));
    th.textContent+=(state.sort.d===1?' ▲':' ▼'); render();});
}

function init(){
  if(!D){document.body.innerHTML='<p style="color:#fff;padding:40px">Data failed to load</p>';return;}
  buildKpis(); buildMatrix(); buildClassAcc(); buildTruePred(); buildEmo();
  fillSel('fTrue',ORDER); fillSel('fPred',ORDER);
  fillSel('fEmo',Object.keys(D.three.emotion_llm||{}));
  const nrcKeys=Object.keys(D.three.emotion_nrc||{});
  fillSel('fNrc',nrcKeys);
  document.getElementById('totCount').textContent=D.rows.length;
  bind(); render();
}
init();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()
