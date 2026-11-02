"""Interactive HTML: soft multi-cluster decomposition of a target activation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Distinct palette for up to 12 clusters (Tableau-like).
CLUSTER_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#86bcb6",
    "#d37295",
]


def write_activation_decomposition_html(
    path: Path,
    *,
    coordinates: np.ndarray,
    explained_variance_ratio: np.ndarray,
    details: list[dict[str, Any]],
    cluster_ids: list[int],
    target_position: int,
    decomposition: list[dict[str, Any]],
    model_id: str,
    layer_index: int,
    keyword: str = "",
    temperature: float = 0.0,
) -> None:
    """Write a Plotly view colored by cluster with a soft-assignment panel."""
    normalized_keyword = keyword.strip().lower()
    color_by_cluster = {
        int(item["cluster_id"]): CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
        for rank, item in enumerate(decomposition)
    }
    rendered_details = []
    for detail, cluster_id in zip(details, cluster_ids, strict=True):
        token_text = str(detail["token"])
        rendered_details.append(
            {
                **detail,
                "cluster_id": int(cluster_id),
                "color": color_by_cluster.get(int(cluster_id), "#7f91a2"),
                "keyword_match": bool(
                    normalized_keyword and normalized_keyword in token_text.strip().lower()
                ),
            }
        )
    rendered_decomposition = [
        {
            **item,
            "color": color_by_cluster[int(item["cluster_id"])],
            "weight_pct": round(100.0 * float(item["weight"]), 2),
        }
        for item in decomposition
    ]
    data = {
        "model": model_id,
        "layer": layer_index,
        "reference_count": len(coordinates) - 1,
        "explained": np.asarray(explained_variance_ratio, dtype=float).tolist(),
        "x": coordinates[:, 0].tolist(),
        "y": coordinates[:, 1].tolist(),
        "z": coordinates[:, 2].tolist(),
        "details": rendered_details,
        "target_position": int(target_position),
        "decomposition": rendered_decomposition,
        "keyword": normalized_keyword,
        "temperature": float(temperature),
        "hard_cluster": int(decomposition[0]["cluster_id"]) if decomposition else -1,
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )
    path.write_text(_HTML.replace("__DATA__", encoded), encoding="utf-8")


# Backward-compatible alias used by older tests / callers.
def write_cluster_pca_html(
    path: Path,
    *,
    coordinates: np.ndarray,
    explained_variance_ratio: np.ndarray,
    details: list[dict[str, Any]],
    target_position: int,
    model_id: str,
    layer_index: int,
    cluster_id: int,
    keyword: str = "fri",
) -> None:
    cluster_ids = [int(cluster_id)] * len(details)
    decomposition = [
        {
            "cluster_id": int(cluster_id),
            "weight": 1.0,
            "distance": 0.0,
            "rank": 1,
            "n_shown": max(0, len(details) - 1),
            "n_total": max(0, len(details) - 1),
            "sample_tokens": [],
        }
    ]
    write_activation_decomposition_html(
        path,
        coordinates=coordinates,
        explained_variance_ratio=explained_variance_ratio,
        details=details,
        cluster_ids=cluster_ids,
        target_position=target_position,
        decomposition=decomposition,
        model_id=model_id,
        layer_index=layer_index,
        keyword=keyword,
        temperature=0.0,
    )


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Activation Space Explorer · soft cluster decomposition</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;font-size:15px;color:#172431;background:#f5f7fa}
*{box-sizing:border-box}body{margin:0}
header{padding:16px 22px 12px;background:#fff;border-bottom:1px solid #d8e0e8}
h1{font-size:24px;margin:0 0 4px}.sub{font-size:14px;color:#607080}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 460px;height:calc(100vh - 88px);min-height:650px}
#plot{width:100%;height:100%}
.sidebar{background:#fff;border-left:1px solid #d8e0e8;padding:16px;overflow:auto}
.panel{border:1px solid #d8e0e8;border-radius:10px;padding:14px 15px;margin-bottom:14px;color:#596a79;font-size:14px;line-height:1.55;overflow-wrap:anywhere}
.panel b{color:#172431;font-size:15px}
.label{color:#7b8996;text-transform:uppercase;letter-spacing:.07em;font-size:11px;font-weight:700}
.context{margin-top:10px;padding:11px;background:#f6f8fa;border-radius:7px;white-space:pre-wrap;color:#344554;font-size:15px}
.chosen{background:#ffd88a;color:#172431;border-radius:3px;padding:1px 3px;font-weight:650}
.target{border-color:#f0ad4e;background:#fffaf0}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.metric{padding:9px;background:#f6f8fa;border-radius:7px}
.metric strong{display:block;color:#172431;font-size:16px}
button{border:1px solid #ccd6df;background:#fff;border-radius:7px;padding:8px 11px;color:#263746;cursor:pointer;font-size:13px}
button:hover{background:#f4f7f9}
.decomp-row{display:grid;grid-template-columns:18px 1fr auto;gap:8px;align-items:center;padding:8px 6px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.decomp-row:hover,.decomp-row.active{background:#f3f7fb;border-color:#d5e2ef}
.swatch{width:14px;height:14px;border-radius:4px;border:1px solid rgba(0,0,0,.12)}
.bar-track{height:8px;background:#e8eef3;border-radius:99px;overflow:hidden;margin-top:4px}
.bar-fill{height:100%;border-radius:99px}
.tokens{margin-top:4px;color:#7a8794;font-size:12px}
.formula{font-size:12px;color:#708090;background:#f6f8fa;border-radius:7px;padding:10px;margin-top:8px}
@media(max-width:960px){.workspace{grid-template-columns:1fr;grid-template-rows:62vh auto;height:auto}.sidebar{border-left:0;border-top:1px solid #d8e0e8}}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="workspace">
  <div id="plot"></div>
  <aside class="sidebar">
    <div class="panel target" id="target"></div>
    <div class="panel" id="decomposition"></div>
    <div class="panel" id="selection"><b>Selected activation</b><br>Click a point or a decomposition row.</div>
    <div class="panel" id="guide"></div>
  </aside>
</div>
<script>
const D=__DATA__;
const pct=x=>(100*x).toFixed(2)+'%';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const target=D.details[D.target_position];
const sumVar=D.explained.reduce((a,b)=>a+b,0);
document.getElementById('title').textContent=`${D.model} · layer ${D.layer} · soft decomposition`;
document.getElementById('subtitle').textContent=
  `${D.reference_count.toLocaleString()} sampled neighbors from top-${D.decomposition.length} clusters + target · joint PCA-3 explains ${pct(sumVar)} · hard assign = cluster ${D.hard_cluster}`;

const contextHtml=d=>`${esc(d.before)}<span class="chosen">${esc(d.token)||'∅'}</span>${esc(d.after)}`;
const display=x=>typeof x==='number'?x.toLocaleString():esc(x);
const detailHtml=(heading,d)=>`<b>${heading}</b>
  <div class="metrics">
    <div class="metric"><span class="label">Cluster</span><strong style="color:${d.color||'#172431'}">${d.cluster_id}</strong></div>
    <div class="metric"><span class="label">Token ID</span><strong>${d.token_id}</strong></div>
    <div class="metric"><span class="label">Row</span><strong>${display(d.row)}</strong></div>
    <div class="metric"><span class="label">Position</span><strong>${display(d.position)}</strong></div>
  </div>
  <div class="context">${contextHtml(d)}</div>`;

document.getElementById('target').innerHTML=detailHtml('Target activation',target)+
  `<div class="formula">Soft weights use <code>softmax(−‖x−cₖ‖ / τ)</code> over all ${D.decomposition.length? 'k-means':'k-means'} centroids.
   Shown τ = ${D.temperature.toFixed(4)}. Bars are the top clusters that “explain” this activation as graded membership (not causal features).</div>`;

let focusCluster=null;
function renderDecomposition(){
  const rows=D.decomposition.map((item,idx)=>{
    const active=focusCluster===item.cluster_id?' active':'';
    const samples=(item.sample_tokens||[]).slice(0,8).map(t=>esc(JSON.stringify(t))).join(' · ');
    return `<div class="decomp-row${active}" data-cluster="${item.cluster_id}">
      <div class="swatch" style="background:${item.color}"></div>
      <div>
        <div><b>#${item.rank} · cluster ${item.cluster_id}</b> · dist ${item.distance.toFixed(3)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${item.weight_pct}%;background:${item.color}"></div></div>
        <div class="tokens">${item.n_shown.toLocaleString()}/${item.n_total.toLocaleString()} shown${samples?` · e.g. ${samples}`:''}</div>
      </div>
      <div><b>${item.weight_pct.toFixed(1)}%</b></div>
    </div>`;
  }).join('');
  document.getElementById('decomposition').innerHTML=
    `<div class="label">Graded cluster decomposition</div>${rows || '<div>No clusters</div>'}`;
  document.querySelectorAll('.decomp-row').forEach(el=>{
    el.onclick=()=>{
      const cid=Number(el.dataset.cluster);
      focusCluster = focusCluster===cid ? null : cid;
      renderDecomposition();
      restylePlot();
      const item=D.decomposition.find(d=>d.cluster_id===cid);
      if(item){
        document.getElementById('selection').innerHTML=
          `<b>Cluster ${cid}</b><br>Soft weight <b>${item.weight_pct.toFixed(2)}%</b> · distance ${item.distance.toFixed(3)}<br>
           Sample tokens: ${(item.sample_tokens||[]).slice(0,16).map(t=>esc(JSON.stringify(t))).join(', ') || '—'}`;
      }
    };
  });
}

const byCluster={};
D.details.forEach((d,i)=>{
  if(i===D.target_position) return;
  (byCluster[d.cluster_id] ||= []).push(i);
});

function buildTraces(){
  const traces=[];
  D.decomposition.forEach(item=>{
    const indices=byCluster[item.cluster_id]||[];
    if(!indices.length) return;
    const dimmed=focusCluster!==null && focusCluster!==item.cluster_id;
    traces.push({
      type:'scatter3d', mode:'markers', name:`c${item.cluster_id} (${item.weight_pct.toFixed(1)}%)`,
      showlegend:true,
      x:indices.map(i=>D.x[i]), y:indices.map(i=>D.y[i]), z:indices.map(i=>D.z[i]),
      customdata:indices,
      text:indices.map(i=>D.details[i].row),
      hovertext:indices.map(i=>D.details[i].token),
      marker:{
        size: dimmed?2:4,
        color:item.color,
        opacity: dimmed?0.08:0.75,
        line:{width:0}
      },
      hovertemplate:`cluster ${item.cluster_id}<br>row %{text}<br>token %{hovertext}<extra>${item.weight_pct.toFixed(1)}%</extra>`
    });
  });
  traces.push({
    type:'scatter3d', mode:'markers', name:'Target', showlegend:true,
    x:[D.x[D.target_position]], y:[D.y[D.target_position]], z:[D.z[D.target_position]],
    customdata:[D.target_position],
    marker:{size:11, color:'#111827', symbol:'diamond',
      line:{color:'#f8fafc', width:2}, opacity:1},
    hovertemplate:`TARGET<br>hard cluster ${D.hard_cluster}<extra></extra>`
  });
  return traces;
}

const axis=(name,value)=>({title:`${name} · ${pct(value)}`, backgroundcolor:'#fff', gridcolor:'#dce4eb', showbackground:true, zerolinecolor:'#aebbc7'});
const layout={
  margin:{l:0,r:0,t:0,b:0}, paper_bgcolor:'#f5f7fa',
  font:{family:'Inter, ui-sans-serif, system-ui, sans-serif', size:13, color:'#263746'},
  legend:{x:0.01,y:0.99,font:{size:12},bgcolor:'rgba(255,255,255,.9)'},
  scene:{xaxis:axis('PC1',D.explained[0]), yaxis:axis('PC2',D.explained[1]), zaxis:axis('PC3',D.explained[2]), aspectmode:'data'}
};
const plotConfig={responsive:true, displaylogo:false, scrollZoom:true};
let plotNode=null;

function restylePlot(){
  if(!plotNode) return;
  Plotly.react(plotNode, buildTraces(), layout, plotConfig);
}

document.getElementById('guide').innerHTML=`
  <div class="label">How to read</div>
  <p>Each color is one of the <b>top soft-assigned k-means clusters</b> for the target activation.
  Bar length ≈ how much that cluster contributes under a distance-based softmax (graded membership).
  Joint PCA is fit on <b>L2-normalized</b> sampled neighbors (direction, not magnitude); the black diamond is the target in that space.
  Prompts are forwarded with an attention-sink prefix so the first user token is not absolute position 0.</p>
  <p>${D.keyword?`Keyword highlight is available in point hover/context when tokens contain <b>${esc(D.keyword)}</b>.`:``}</p>
  <button id="reset">Reset camera</button>
  <button id="clearFocus">Show all clusters</button>`;

renderDecomposition();
Plotly.newPlot('plot', buildTraces(), layout, plotConfig).then(plot=>{
  plotNode=plot;
  plot.on('plotly_click', event=>{
    const point=event.points&&event.points[0];
    if(!point) return;
    const idx=point.customdata;
    const d=D.details[idx];
    document.getElementById('selection').innerHTML=
      detailHtml(idx===D.target_position?'Target activation':`Neighbor · cluster ${d.cluster_id}`, d);
  });
  document.getElementById('reset').onclick=()=>Plotly.relayout(plot,{'scene.camera':null});
  document.getElementById('clearFocus').onclick=()=>{focusCluster=null; renderDecomposition(); restylePlot();};
});
</script>
</body>
</html>"""
